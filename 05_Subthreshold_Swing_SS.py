# =============================================================================
# PARTE IV - SUBTHRESHOLD SWING COM MARGENS DE ERRO (SEM SCIPY)
# =============================================================================

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class AnalisadorSSRobusto:
    """Classe para análise robusta do SS com estimativa de erros - SEM SCIPY"""
    
    @staticmethod
    def aplicar_savitzky_golay_com_incerteza(y, window_size=11, polyorder=3):
        """
        Savitzky-Golay com estimativa de incerteza - SEM SCIPY
        """
        try:
            if len(y) < window_size:
                return y, np.zeros_like(y)
            
            if window_size % 2 == 0:
                window_size += 1
            
            half_window = window_size // 2
            y_smooth = np.zeros_like(y)
            y_std = np.zeros_like(y)
            
            for i in range(len(y)):
                start = max(0, i - half_window)
                end = min(len(y), i + half_window + 1)
                window_data = y[start:end]
                
                if len(window_data) >= polyorder + 1:
                    x_window = np.arange(len(window_data))
                    coeffs = np.polyfit(x_window, window_data, polyorder)
                    y_smooth[i] = np.polyval(coeffs, half_window)
                    
                    # Calcular desvio padrão residual como estimativa de erro
                    y_pred = np.polyval(coeffs, x_window)
                    residuals = window_data - y_pred
                    y_std[i] = np.std(residuals)
                else:
                    y_smooth[i] = y[i]
                    y_std[i] = 0
            
            return y_smooth, y_std
            
        except Exception as e:
            print(f"⚠️  Erro na suavização: {e}")
            return y, np.zeros_like(y)
    
    @staticmethod
    def calcular_derivada_com_erro(y, x, y_std=None):
        """
        Calcula derivada com propagação de erro - SEM SCIPY
        """
        dy_dx = np.gradient(y, x)
        
        if y_std is not None and np.any(y_std > 0):
            # Estimativa grosseira do erro na derivada
            dx = np.mean(np.diff(x))
            dy_std = y_std * np.sqrt(2) / dx  # Propagação de erro simplificada
        else:
            dy_std = np.zeros_like(dy_dx)
        
        return dy_dx, dy_std
    
    @staticmethod
    def calcular_intervalo_confianca_manual(erro, nivel_confianca=0.95):
        """
        Calcula intervalo de confiança manualmente - SEM SCIPY
        """
        # Valores Z aproximados para nível de confiança
        z_scores = {
            0.90: 1.645,
            0.95: 1.960,
            0.99: 2.576
        }
        
        z_score = z_scores.get(nivel_confianca, 1.960)  # Default 95%
        return z_score * erro

def calcular_SS_com_erro(df_ida, nome_arquivo="", window_size=11, nivel_confianca=0.95, mostrar_grafico=True):
    """
    Cálculo do SS com estimativa de margens de erro - SEM SCIPY
    """
    
    print("🔍 PARTE IV - SS COM ANÁLISE DE ERRO (SEM SCIPY)")
    print("=" * 60)
    
    # ETAPA 1: PREPARAÇÃO DOS DADOS
    vgs_orig = df_ida['BaseV'].values
    Id_orig = df_ida['CollectorI'].values
    
    mask_positivo = Id_orig > 1e-12
    vgs = vgs_orig[mask_positivo]
    Id = Id_orig[mask_positivo]
    
    sort_idx = np.argsort(vgs)
    vgs = vgs[sort_idx]
    Id = Id[sort_idx]
    
    print(f"1️⃣ Dados preparados: {len(vgs)} pontos")
    
    # ETAPA 2: ESCALA LOG E SUAVIZAÇÃO COM ERRO
    log_Id = np.log10(Id)
    log_Id_smooth, log_Id_std = AnalisadorSSRobusto.aplicar_savitzky_golay_com_incerteza(
        log_Id, window_size=window_size, polyorder=3
    )
    
    erro_medio_log = np.mean(log_Id_std[log_Id_std > 0]) if np.any(log_Id_std > 0) else 0
    print(f"2️⃣ Suavização com estimativa de erro")
    print(f"   Erro médio log(Id): {erro_medio_log:.4f}")
    
    # ETAPA 3: DERIVADA COM PROPAGAÇÃO DE ERRO
    dlogId_dVgs, derivada_erro = AnalisadorSSRobusto.calcular_derivada_com_erro(
        log_Id_smooth, vgs, log_Id_std
    )
    
    erro_medio_derivada = np.mean(derivada_erro[derivada_erro > 0]) if np.any(derivada_erro > 0) else 0
    print(f"3️⃣ Derivada com propagação de erro")
    print(f"   Erro médio derivada: {erro_medio_derivada:.4f}")
    
    # ETAPA 4: SS COM INTERVALO DE CONFIANÇA
    SS = np.zeros_like(dlogId_dVgs)
    SS_erro = np.zeros_like(dlogId_dVgs)
    
    for i in range(len(dlogId_dVgs)):
        if dlogId_dVgs[i] > 1e-6:  # Evitar divisão por zero
            SS[i] = (1 / dlogId_dVgs[i]) * 1000  # mV/dec
            # Propagação de erro: d(SS) = |d(1/x)| * dx = (1/x²) * dx
            SS_erro[i] = (1 / (dlogId_dVgs[i]**2)) * derivada_erro[i] * 1000
        else:
            SS[i] = float('inf')
            SS_erro[i] = float('inf')
    
    # Calcular intervalos de confiança manualmente
    SS_intervalo = AnalisadorSSRobusto.calcular_intervalo_confianca_manual(SS_erro, nivel_confianca)
    SS_inferior = np.maximum(60, SS - SS_intervalo)  # Mínimo 60 mV/dec
    SS_superior = SS + SS_intervalo
    
    # ETAPA 5: FILTRAGEM ROBUSTA
    # Critério: erro < 50% do valor e SS dentro de faixa física
    mask_fisico = (SS >= 60) & (SS <= 2000) & (SS_erro < SS * 0.5) & (SS_erro > 0)
    
    if np.any(mask_fisico):
        SS_filtrado = SS[mask_fisico]
        SS_erro_filtrado = SS_erro[mask_fisico]
        SS_inf_filtrado = SS_inferior[mask_fisico]
        SS_sup_filtrado = SS_superior[mask_fisico]
        vgs_filtrado = vgs[mask_fisico]
        log_Id_filtrado = log_Id_smooth[mask_fisico]
        
        # Encontrar SS mínimo considerando incerteza
        idx_SS_min = np.argmin(SS_filtrado)
        SS_min = SS_filtrado[idx_SS_min]
        SS_min_erro = SS_erro_filtrado[idx_SS_min]
        Vgs_SS_min = vgs_filtrado[idx_SS_min]
        log_Id_SS_min = log_Id_filtrado[idx_SS_min]
        
        # Estatísticas robustas
        SS_medio = np.mean(SS_filtrado)
        SS_mediano = np.median(SS_filtrado)
        SS_std = np.std(SS_filtrado)
        
        # Coeficiente de variação como medida de qualidade
        coef_variacao = (SS_std / SS_medio) * 100 if SS_medio > 0 else float('inf')
        
        print(f"4️⃣ SS mínimo: {SS_min:.3f} ± {SS_min_erro:.3f} mV/dec")
        print(f"   Coef. variação: {coef_variacao:.1f}%")
        print(f"   Pontos confiáveis: {len(SS_filtrado)}")
        
    else:
        raise ValueError("Nenhum ponto confiável encontrado!")
    
    # ETAPA 6: ANÁLISE VISUAL COM ERROS
    if mostrar_grafico:
        plotar_analise_SS_com_erro(vgs, Id, log_Id, log_Id_smooth, log_Id_std,
                                 vgs_filtrado, SS_filtrado, SS_erro_filtrado,
                                 SS_min, SS_min_erro, Vgs_SS_min, log_Id_SS_min,
                                 nivel_confianca, nome_arquivo)
    
    # RESULTADOS COM MARGENS DE ERRO
    resultados = {
        'SS_min': SS_min,
        'SS_min_erro': SS_min_erro,
        'SS_min_intervalo': (SS_min - SS_min_erro, SS_min + SS_min_erro),
        'Vgs_SS_min': Vgs_SS_min,
        'log_Id_SS_min': log_Id_SS_min,
        'SS_medio': SS_medio,
        'SS_mediano': SS_mediano,
        'SS_std': SS_std,
        'coef_variacao': coef_variacao,
        'pontos_confiaveis': len(SS_filtrado),
        'pontos_totais': len(SS),
        'nivel_confianca': nivel_confianca,
        'qualidade_ajuste': 'Excelente' if coef_variacao < 20 else 
                           'Bom' if coef_variacao < 40 else 'Moderado',
        'nome_arquivo': nome_arquivo
    }
    
    return resultados

def plotar_analise_SS_com_erro(vgs, Id, log_Id, log_Id_smooth, log_Id_std,
                             vgs_SS, SS, SS_erro, SS_min, SS_min_erro, 
                             Vgs_SS_min, log_Id_SS_min, nivel_confianca, nome_arquivo):
    """
    Plotagem com visualização de erros e intervalos de confiança - SEM SCIPY
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico 1: Curva com banda de incerteza
    ax1.semilogy(vgs, Id, 'ko', alpha=0.2, markersize=2, label='I_D original')
    ax1.semilogy(vgs, 10**log_Id_smooth, 'r-', linewidth=2, 
                label='I_D suavizado')
    
    # Banda de incerteza (convertendo de log para linear)
    if np.any(log_Id_std > 0):
        incerteza_superior = 10**(log_Id_smooth + log_Id_std)
        incerteza_inferior = 10**(log_Id_smooth - log_Id_std)
        ax1.fill_between(vgs, incerteza_inferior, incerteza_superior, 
                        alpha=0.3, color='red', label='Incerteza suavização')
    
    ax1.axvline(Vgs_SS_min, color='blue', linestyle='--', alpha=0.7,
                label=f'Vgs @ SS_min')
    ax1.set_xlabel('V_GS (V)')
    ax1.set_ylabel('I_D (A)')
    ax1.set_title('1. Curva com Incerteza da Suavização')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: SS com barras de erro
    ax2.errorbar(vgs_SS, SS, yerr=SS_erro, fmt='go-', alpha=0.7,
                markersize=3, linewidth=1, capsize=3, capthick=1,
                label='SS com erro')
    ax2.plot(Vgs_SS_min, SS_min, 'ro', markersize=8, 
             label=f'SS_min = {SS_min:.1f} ± {SS_min_erro:.1f} mV/dec')
    
    ax2.axhline(60, color='orange', linestyle='--', alpha=0.7, 
                label='Limite teórico 60 mV/dec')
    ax2.axhline(100, color='red', linestyle='--', alpha=0.5, 
                label='Bom (<100 mV/dec)')
    
    ax2.set_xlabel('V_GS (V)')
    ax2.set_ylabel('Subthreshold Swing (mV/dec)')
    ax2.set_title('2. SS com Barras de Erro')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Ajustar limites do eixo Y
    y_max = min(500, np.max(SS + SS_erro) * 1.2) if len(SS) > 0 else 500
    ax2.set_ylim(0, y_max)
    
    # Gráfico 3: Histograma com estatísticas
    if len(SS) > 0:
        ax3.hist(SS, bins=min(15, len(SS)//3), alpha=0.7, color='purple', edgecolor='black',
                density=True)
        ax3.axvline(SS_min, color='red', linestyle='--', linewidth=2, 
                    label=f'SS_min = {SS_min:.1f} mV/dec')
        ax3.axvline(np.mean(SS), color='blue', linestyle='--', alpha=0.7,
                    label=f'Média = {np.mean(SS):.1f} mV/dec')
        ax3.axvline(np.median(SS), color='green', linestyle='--', alpha=0.7,
                    label=f'Mediana = {np.median(SS):.1f} mV/dec')
        
        ax3.set_xlabel('Subthreshold Swing (mV/dec)')
        ax3.set_ylabel('Densidade de Probabilidade')
        ax3.set_title('3. Distribuição do SS\n(Análise Estatística)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Dados insuficientes\npara histograma', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('3. Distribuição do SS\n(Dados Insuficientes)')
    
    # Gráfico 4: Resumo de qualidade COM NOME DO ARQUIVO
    ax4.axis('off')
    
    # Texto com análise de qualidade incluindo nome do arquivo
    texto_analise = (
        f'📋 ANÁLISE DE QUALIDADE - SS\n\n'
        f'📁 Arquivo: {nome_arquivo}\n\n'
        f'🎯 SS mínimo: {SS_min:.1f} ± {SS_min_erro:.1f} mV/dec\n'
        f'📊 Intervalo: [{SS_min-SS_min_erro:.1f}, {SS_min+SS_min_erro:.1f}] mV/dec\n\n'
        f'📈 ESTATÍSTICAS:\n'
        f'Média: {np.mean(SS):.1f} mV/dec\n'
        f'Mediana: {np.median(SS):.1f} mV/dec\n'
        f'Desvio Padrão: {np.std(SS):.1f} mV/dec\n'
        f'Coef. Variação: {(np.std(SS)/np.mean(SS)*100):.1f}%\n\n'
        f'📊 AMOSTRA:\n'
        f'Pontos confiáveis: {len(SS)}\n'
        f'Vgs @ SS_min: {Vgs_SS_min:.3f} V\n'
        f'Nível confiança: {nivel_confianca*100:.0f}%'
    )
    
    # Avaliação qualitativa
    if SS_min <= 80:
        avaliacao = "🎯 EXCELENTE"
        cor_avaliacao = "green"
    elif SS_min <= 120:
        avaliacao = "✅ BOM" 
        cor_avaliacao = "blue"
    elif SS_min <= 200:
        avaliacao = "⚠️  MODERADO"
        cor_avaliacao = "orange"
    else:
        avaliacao = "❌ ALTO (ruído/interface)"
        cor_avaliacao = "red"
    
    ax4.text(0.05, 0.95, texto_analise, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace', linespacing=1.4,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    ax4.text(0.05, 0.25, avaliacao, transform=ax4.transAxes, fontsize=14,
             verticalalignment='top', fontweight='bold', color=cor_avaliacao,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.show()

class InterfaceSSRobusto:
    def __init__(self, root):
        self.root = root
        self.root.title("Parte IV - SS com Análise de Erro")
        self.root.geometry("600x500")
        
        self.df_ida = None
        self.nome_arquivo = ""
        self.criar_interface()
    
    def criar_interface(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        titulo = ttk.Label(main_frame, 
                          text="🧮 Parte IV - Subthreshold Swing\nAnálise Robusta com Margens de Erro", 
                          font=('Arial', 14, 'bold'), justify=tk.CENTER)
        titulo.grid(row=0, column=0, pady=10)
        
        # Explicação das melhorias
        melhorias_text = """
• Propagação de erro na suavização e derivada
• Intervalos de confiança para o SS
• Filtragem por qualidade dos pontos
• Análise estatística robusta
• Avaliação automática de qualidade
• ✅ SEM dependência do scipy
• 📁 Nome do arquivo incluído no relatório
        """
        
        melhorias_label = ttk.Label(main_frame, text=melhorias_text, 
                                   font=('Arial', 9), justify=tk.LEFT)
        melhorias_label.grid(row=1, column=0, pady=10)
        
        # Botão carregar arquivo
        btn_carregar = ttk.Button(main_frame, text="1. Carregar Arquivo CSV", 
                                 command=self.carregar_arquivo)
        btn_carregar.grid(row=2, column=0, pady=10, sticky=tk.EW)
        
        # Frame configurações
        config_frame = ttk.LabelFrame(main_frame, text="Configurações de Análise", padding="10")
        config_frame.grid(row=3, column=0, pady=10, sticky=tk.EW)
        
        # Janela
        ttk.Label(config_frame, text="Janela suavização:").grid(row=0, column=0, sticky=tk.W)
        self.janela_var = tk.StringVar(value="11")
        janela_combo = ttk.Combobox(config_frame, values=["7", "9", "11", "13", "15"],
                                   textvariable=self.janela_var, state="readonly", width=8)
        janela_combo.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Nível de confiança
        ttk.Label(config_frame, text="Nível confiança:").grid(row=1, column=0, sticky=tk.W)
        self.confianca_var = tk.StringVar(value="0.95")
        confianca_combo = ttk.Combobox(config_frame, values=["0.90", "0.95", "0.99"],
                                      textvariable=self.confianca_var, state="readonly", width=8)
        confianca_combo.grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # Label status
        self.status_label = ttk.Label(main_frame, text="Nenhum arquivo carregado", 
                                     foreground="red", wraplength=500)
        self.status_label.grid(row=4, column=0, pady=10)
        
        # Botão calcular
        btn_calcular = ttk.Button(main_frame, text="2. Análise Robusta do SS", 
                                 command=self.analisar_SS_robusto)
        btn_calcular.grid(row=5, column=0, pady=15, sticky=tk.EW)
        
        # Configurar expansão
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
    
    def carregar_arquivo(self):
        arquivo = filedialog.askopenfilename(
            title="Selecionar arquivo CSV",
            filetypes=[("CSV files", "*.csv"), ("Todos os arquivos", "*.*")]
        )
        
        if arquivo:
            try:
                df = pd.read_csv(arquivo, delimiter=';', decimal=',')
                df = df.iloc[:, :2]
                df.columns = ['BaseV', 'CollectorI']
                df = df.apply(pd.to_numeric, errors='coerce').dropna()
                
                n_total = len(df)
                n_ida = n_total // 2
                self.df_ida = df.iloc[:n_ida].copy().sort_values('BaseV')
                self.nome_arquivo = os.path.basename(arquivo)
                
                self.status_label.config(
                    text=f"✅ Arquivo: {self.nome_arquivo}\n"
                         f"📊 {len(self.df_ida)} pontos | Pronto para análise robusta", 
                    foreground="green"
                )
                
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao carregar:\n{str(e)}")
                self.status_label.config(text="❌ Erro ao carregar arquivo", foreground="red")
    
    def analisar_SS_robusto(self):
        if self.df_ida is None:
            messagebox.showwarning("Aviso", "Carregue um arquivo CSV primeiro!")
            return
        
        try:
            janela = int(self.janela_var.get())
            confianca = float(self.confianca_var.get())
            
            print("🚀 INICIANDO ANÁLISE ROBUSTA DO SS (SEM SCIPY)")
            print("=" * 60)
            
            resultados = calcular_SS_com_erro(self.df_ida, self.nome_arquivo, window_size=janela, 
                                            nivel_confianca=confianca)
            
            # Formatar resultado com margem de erro
            SS_formatado = f"{resultados['SS_min']:.1f} ± {resultados['SS_min_erro']:.1f}"
            
            resumo = (
                f"📋 RESULTADOS - ANÁLISE ROBUSTA\n\n"
                f"📁 Arquivo: {self.nome_arquivo}\n\n"
                f"🎯 SS mínimo = {SS_formatado} mV/dec\n"
                f"📊 Intervalo: {resultados['SS_min_intervalo'][0]:.1f} - {resultados['SS_min_intervalo'][1]:.1f} mV/dec\n\n"
                f"📈 Qualidade: {resultados['qualidade_ajuste']}\n"
                f"📏 Coef. variação: {resultados['coef_variacao']:.1f}%\n"
                f"🔢 Pontos confiáveis: {resultados['pontos_confiaveis']}\n\n"
                f"⚡ V_GS @ SS_min: {resultados['Vgs_SS_min']:.3f} V\n"
                f"🎯 Nível confiança: {confianca*100:.0f}%\n\n"
                f"✅ Análise com margens de erro concluída!"
            )
            
            messagebox.showinfo("SS - Análise Robusta", resumo)
            
            # Atualizar status com cor baseada na qualidade
            if resultados['qualidade_ajuste'] == 'Excelente':
                cor = "green"
            elif resultados['qualidade_ajuste'] == 'Bom':
                cor = "blue"
            else:
                cor = "orange"
                
            self.status_label.config(text=f"✅ SS: {SS_formatado} mV/dec ({resultados['qualidade_ajuste']})", 
                                   foreground=cor)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na análise:\n{str(e)}")
            self.status_label.config(text=f"❌ {str(e)}", foreground="red")

if __name__ == "__main__":
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        
        root = tk.Tk()
        app = InterfaceSSRobusto(root)
        root.mainloop()
        
    except ImportError as e:
        print(f"Instale: pip install pandas numpy matplotlib")