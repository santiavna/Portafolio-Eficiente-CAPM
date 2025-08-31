import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy.optimize as opt
from scipy.stats import jarque_bera

# ------------------------------------------------------------------
# Configuración inicial
st.title("Panel de ADRs y Portafolios")

# Selector general de sección
seccion = st.sidebar.selectbox("Sección", ["Portafolios", "CAPM & VaR"])

# Tickers + nombre descriptivo
tickers_info = {
    "BBAR": "Banco BBVA Argentina",
    "BMA": "Banco Macro",
    "CEPU": "Central Puerto",
    "CRESY": "Cresud",
    "EDN": "Edenor",
    "GGAL": "Grupo Financiero Galicia",
    "IRS": "IRSA",
    "LOMA": "Loma Negra",
    "MELI": "Mercado Libre",
    "PAM": "Pampa Energía",
    "SUPV": "Grupo Supervielle",
    "TEO": "Telecom Argentina",
    "TGS": "Transportadora de Gas del Sur",
    "TS": "Tenaris",
    "YPF": "YPF"
}

# Selector de rango de fechas
st.sidebar.subheader("Seleccionar período")
date_min = datetime.date(2020, 1, 2)
date_max = datetime.date(2025, 7, 25)
date_ini = st.sidebar.date_input("Desde", date_min, min_value=date_min, max_value=date_max)
date_end = st.sidebar.date_input("Hasta", date_max, min_value=date_min, max_value=date_max)

# Selector de activos
st.sidebar.subheader("Seleccionar activos")
options_list = [f"{t} - {name}" for t, name in tickers_info.items()]

if "selected_all" not in st.session_state:
    st.session_state.selected_all = True

if st.sidebar.button("Marcar / desmarcar todos"):
    st.session_state.selected_all = not st.session_state.selected_all

selected = []
for option in options_list:
    initial = True if st.session_state.selected_all else False
    if st.sidebar.checkbox(option, value=initial, key=option):
        selected.append(option)

selected_tickers = [item.split(" - ")[0] for item in selected]

# Selector de tasa libre de riesgo (solo para CAPM)
rf_choice = st.sidebar.selectbox(
    "Tasa libre de riesgo",
    [0.04, 0.045, 0.05],
    format_func=lambda x: f"{x*100:.1f}%"
)

# ------------------------------------------------------------------

# --- al principio del archivo (después de los imports) ---
if "ready" not in st.session_state:
    st.session_state.ready = False

def run_update():
    st.session_state.ready = True

# --- en la sidebar ---
st.sidebar.button("Actualizar", on_click=run_update)

# --- antes ibas a poner todo lo demás dentro de este if ---
if st.session_state.ready:

    if not selected_tickers:
        st.warning("Seleccioná al menos un activo.")

    else:
        # ---------------------------------------- PORTAFOLIOS
        if seccion == "Portafolios":

            prices = yf.download(selected_tickers, start=date_ini, end=date_end)
            prices = prices.dropna(axis=1, how='all')
            prices = prices.xs('Close', level=0, axis=1)

            prices_norm = prices / prices.iloc[0] * 100
            if isinstance(prices_norm.columns, pd.MultiIndex):
                prices_norm = prices_norm.droplevel(0, axis=1)

            st.subheader("Gráfico de precios indexados (100)")
            st.line_chart(prices_norm)

            # -------- Heatmap de correlaciones
            log_returns = np.log(prices / prices.shift(1)).dropna()
            st.subheader("Matriz de correlaciones entre activos")
            corr_matrix = log_returns.corr()
            fig, ax = plt.subplots()
            cax = ax.imshow(corr_matrix, vmin=-1, vmax=1)
            fig.colorbar(cax)
            ax.set_xticks(range(len(corr_matrix.columns)))
            ax.set_yticks(range(len(corr_matrix.index)))
            ax.set_xticklabels(corr_matrix.columns, rotation=90)
            ax.set_yticklabels(corr_matrix.index)
            st.pyplot(fig)


            # ----- Matriz de varianzas y covarianzas
            cov_matrix = log_returns.cov()
            st.subheader("Matriz de varianzas y covarianzas")
            st.dataframe(cov_matrix.style.format("{:.6f}"))

            # -------- Estadísticas básicas (diarias y anualizadas)
            medios = log_returns.mean()          # media diaria
            varianzas = log_returns.var()        # varianza diaria
            desvios = log_returns.std()          # desvío diario

            # Coeficiente de variación (σ / μ) - diario
            coef_var = desvios / medios

            # Tasa libre de riesgo
            rf_daily = rf_choice / 252

            # Sharpe diario (con rf diaria)
            sharpe_daily = (medios - rf_daily) / desvios

            # Anualización
            media_anual = medios * 252
            desvio_anual = desvios * np.sqrt(252)
            var_anual = varianzas * 252

            # Sharpe anual (con rf anual)
            sharpe_anual = (media_anual - rf_choice) / desvio_anual

            # Tabla combinada
            stats_df = pd.DataFrame({
                "media_d": medios,
                "var_d": varianzas,
                "desvio_d": desvios,
                "coef_var_d": coef_var,
                "sharpe_d": sharpe_daily,
                "media_a": media_anual,
                "var_a": var_anual,
                "desvio_a": desvio_anual,
                "sharpe_a": sharpe_anual
            })

            st.subheader("Estadísticas básicas por activo")
            st.caption("d = diario, a = anual. Sharpe calculado con tasa libre de riesgo seleccionada.")
            st.dataframe(
                stats_df.style.format({
                    "media_d": "{:.4%}", "var_d": "{:.6f}", "desvio_d": "{:.4%}",
                    "coef_var_d": "{:.2f}", "sharpe_d": "{:.2f}",
                    "media_a": "{:.2%}", "var_a": "{:.6f}", "desvio_a": "{:.2%}",
                    "sharpe_a": "{:.2f}"
                })
            )


            # ----- Activo más volátil
            vol_d = log_returns.std()                  # volatilidad diaria
            vol_a = vol_d * np.sqrt(252)               # volatilidad anualizada

            top_ticker_d = vol_d.idxmax()
            top_ticker_a = vol_a.idxmax()

            st.subheader("Activo más volátil")
            st.write(f"Diario: **{top_ticker_d}** ({vol_d[top_ticker_d]:.2%})")
            st.write(f"Anual: **{top_ticker_a}** ({vol_a[top_ticker_a]:.2%})")

            # Top 5 volatilidad anualizada
            top5 = vol_a.sort_values(ascending=False).head(5).to_frame("Volatilidad anualizada")
            st.subheader("Top 5 activos más volátiles (anual)")
            st.dataframe(top5.style.format("{:.2%}"))

            # -------- Portafolios óptimos
            cov_matrix = log_returns.cov()
            mean_ret = medios.values
            cov_np = cov_matrix.values
            n = len(mean_ret)

            def port_stats(w):
                mu = np.dot(w, mean_ret)
                sigma = np.sqrt(np.dot(w.T, np.dot(cov_np, w)))
                return mu, sigma

            def var_obj(w): return np.dot(w.T, np.dot(cov_np, w))
            cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
            bounds = tuple((0, 1) for _ in range(n))
            w0 = np.ones(n) / n

            res_min = opt.minimize(var_obj, w0, method="SLSQP",
                                   bounds=bounds, constraints=cons)
            w_min = res_min.x

            # -------- Mínima varianza con retorno anual ≥ 12%
            mu_target = 0.12 / 252  # 12% anual expresado en retorno diario
            cons_target = (
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "ineq", "fun": lambda w: np.dot(w, mean_ret) - mu_target}
            )
            res_min_target = opt.minimize(var_obj, w0, method="SLSQP",
                                          bounds=bounds, constraints=cons_target)
            w_min_target = res_min_target.x


            rf_daily = rf_choice / 252
            def neg_sharpe(w):
                mu, sigma = port_stats(w)
                return -(mu - rf_daily) / sigma


            res_sh = opt.minimize(neg_sharpe, w0, method="SLSQP",
                                  bounds=bounds, constraints=cons)
            w_sh = res_sh.x

            # -------- Máximo Sharpe con retorno anual ≥ 12%
            res_sh_target = opt.minimize(neg_sharpe, w0, method="SLSQP",
                                         bounds=bounds, constraints=cons_target)
            w_sh_target = res_sh_target.x


            def neg_mu(w): return -np.dot(w, mean_ret)
            res_ret = opt.minimize(neg_mu, w0, method="SLSQP",
                                   bounds=bounds, constraints=cons)
            w_ret = res_ret.x

            # ======== Frontera eficiente (sin short selling) ========
            st.subheader("Frontera eficiente (sin short)")

            # Si hay menos de 2 activos, no tiene sentido trazar frontera
            if len(log_returns.columns) < 2:
                st.info("Necesitás al menos 2 activos para trazar la frontera eficiente.")
            else:
                # Función para obtener el portafolio de mínima varianza dado un retorno objetivo
                def min_vol_for_target(mu_target):
                    cons2 = (
                        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                        {"type": "eq", "fun": lambda w: np.dot(w, mean_ret) - mu_target},
                    )
                    res = opt.minimize(var_obj, w0, method="SLSQP", bounds=bounds, constraints=cons2)
                    return res

            # Retornos objetivo entre el retorno del min var y el de máx retorno (en media diaria)
            mu_min = np.dot(w_min, mean_ret)
            mu_max = np.dot(w_ret, mean_ret)
            lo, hi = (min(mu_min, mu_max), max(mu_min, mu_max))
            target_mus = np.linspace(lo, hi, 60)

            # Construir frontera (ignorando puntos que no convergen)
            frontier_sig = []
            frontier_mu = []
            for m in target_mus:
                res = min_vol_for_target(m)
                if res.success:
                    w = res.x
                    mu, sig = port_stats(w)
                    frontier_mu.append(mu)
                    frontier_sig.append(sig)

            # Simulación Monte Carlo (sin short: pesos >=0 y suman 1)
            n = len(mean_ret)
            M = 4000
            rand_w = np.random.dirichlet(np.ones(n), size=M)
            rand_mu = rand_w.dot(mean_ret)
            rand_sig = np.sqrt(np.sum((rand_w @ cov_np) * rand_w, axis=1))

            # Anualización
            def ann_stats(w):
                mu_d, sig_d = port_stats(w)
                return mu_d * 252, sig_d * np.sqrt(252)

            mu_ann_min, sig_ann_min = ann_stats(w_min)
            mu_ann_sh , sig_ann_sh  = ann_stats(w_sh)
            mu_ann_ret, sig_ann_ret = ann_stats(w_ret)

            rand_mu_ann = rand_mu * 252
            rand_sig_ann = rand_sig * np.sqrt(252)
            frontier_mu_ann = np.array(frontier_mu) * 252
            frontier_sig_ann = np.array(frontier_sig) * np.sqrt(252)

            # Gráfico
            fig, ax = plt.subplots(figsize=(7, 5))

            # Nube de carteras simuladas
            ax.scatter(rand_sig_ann, rand_mu_ann, s=8, alpha=0.35, label="Portafolios simulados")

            # Frontera eficiente
            ax.plot(frontier_sig_ann, frontier_mu_ann, lw=2, label="Frontera eficiente")

            # Tres portafolios clave (con marcadores distintos)
            ax.scatter(sig_ann_min, mu_ann_min, s=110, marker="o", edgecolor="k", label="Mín Var")
            ax.scatter(sig_ann_sh , mu_ann_sh , s=140, marker="*", edgecolor="k", label="Máx Sharpe")
            ax.scatter(sig_ann_ret, mu_ann_ret, s=110, marker="^", edgecolor="k", label="Máx Ret")

            # Etiquetas cerca de cada punto para que se identifiquen fácil
            ax.annotate("Mín Var", xy=(sig_ann_min, mu_ann_min), xytext=(8, 8), textcoords="offset points")
            ax.annotate("Máx Sharpe", xy=(sig_ann_sh, mu_ann_sh), xytext=(8, -15), textcoords="offset points")
            ax.annotate("Máx Ret", xy=(sig_ann_ret, mu_ann_ret), xytext=(8, 8), textcoords="offset points")

            # Capital Allocation Line (usa la tasa que elegiste en el sidebar)
            rf_ann = rf_choice
            if sig_ann_sh > 0:
                x_cal = np.linspace(0, sig_ann_sh * 1.3, 50)
                slope = (mu_ann_sh - rf_ann) / sig_ann_sh
                y_cal = rf_ann + slope * x_cal
                ax.plot(x_cal, y_cal, linestyle="--", linewidth=1, label=f"CAL (rf={rf_ann:.1%})")

            ax.set_xlabel("Volatilidad anualizada")
            ax.set_ylabel("Retorno anualizado")
            ax.set_title("Frontera eficiente y carteras óptimas")
            ax.legend()
            ax.grid(True, alpha=0.25)

            st.pyplot(fig)
# ======== Fin frontera ========


            st.subheader("Portafolios óptimos (pesos)")

            weights_df = pd.DataFrame({
                "Mín Var": w_min,
                "Máx Sharpe": w_sh,
                "Máx Ret": w_ret,
                "Mín Var (>=12% anual)": w_min_target,
                "Máx Sharpe (>=12% anual)": w_sh_target
            }, index=log_returns.columns)


            st.dataframe(weights_df.style.format("{:.2%}"))

            csv_bytes = weights_df.to_csv(float_format="%.6f").encode("utf-8")
            st.download_button(
                label="⬇️ Descargar pesos (CSV)",
                data=csv_bytes,
                file_name="pesos_portafolios.csv",
                mime="text/csv"
            )

        # ---------------------------------------- CAPM & VaR
        elif seccion == "CAPM & VaR":

            st.write("Entrando a sección CAPM")

            # Descargar precios de los activos
            prices = yf.download(selected_tickers, start=date_ini, end=date_end)["Close"]

            # Descargar el índice de mercado (S&P500)
            sp500 = yf.download("^GSPC", start=date_ini, end=date_end)["Close"]

            # Log-returns
            log_returns = np.log(prices / prices.shift(1)).dropna()
            log_mkt     = np.log(sp500  / sp500.shift(1)).dropna()

            # Alinear fechas
            common_idx = log_returns.index.intersection(log_mkt.index)
            log_returns = log_returns.loc[common_idx]
            log_mkt     = log_mkt.loc[common_idx]

            alphas = []
            betas = []
            covs = []
            mkt_var = log_mkt.var()
            capm_exp = []
            var_hist = []
            var_param = []

            merged = pd.concat([log_returns, log_mkt], axis=1).dropna()
            log_returns = merged.iloc[:, :-1]
            log_mkt     = merged.iloc[:, -1]

            # ----- Gráfico comparativo de precios (indexados a 100)
            combined = pd.concat([prices, sp500], axis=1)
            combined_norm = combined / combined.iloc[0] * 100
            st.subheader("Evolución comparada (indexada = 100)")
            st.line_chart(combined_norm)

            for ticker in log_returns.columns:

   	        # covarianza con el mercado
                x = log_mkt.values
                y = log_returns[ticker].values
                cov_ri_rm = np.cov(x, y)[0, 1]
                covs.append(cov_ri_rm)

                # regresión (beta y alpha)
                coef  = np.polyfit(x, y, 1)
                beta  = coef[0]
                alpha = coef[1]
                betas.append(beta)
                alphas.append(alpha)

     	        # CAPM expected return
                mkt_mean = log_mkt.mean()
                exp_capm = rf_choice + beta * (mkt_mean - rf_choice)
                capm_exp.append(exp_capm)

                # VaR histórico (5%)
                var_hist.append(np.percentile(log_returns[ticker], 5))

                # VaR paramétrico (5%)
                z = -1.645
                var_param.append(log_returns[ticker].mean() + z * log_returns[ticker].std())

            df_capm = pd.DataFrame({
                "alpha": alphas,
                "beta": betas,
                "Var mercado": mkt_var,
                "Cov(ri,rm)": covs,
                "CAPM Expected Return": capm_exp,
                "VaR histórico (5%)": var_hist,
                "VaR paramétrico (5%)": var_param
            }, index=log_returns.columns)

            st.subheader("Single Index Model / CAPM")
            st.markdown("**Índice de mercado utilizado:** S&P500 (`^GSPC`)")
            st.dataframe(df_capm)

                        # ----- Test de normalidad -----
            st.subheader("Test de normalidad de los retornos")

            jb_results = {}
            for ticker in log_returns.columns:
                                  stat, p_value = jarque_bera(log_returns[ticker])
                                  jb_results[ticker] = {"Estadístico JB": stat, "p-value": p_value}

            jb_df = pd.DataFrame(jb_results).T

            st.dataframe(jb_df.style.format({"Estadístico JB": "{:.2f}", "p-value": "{:.4f}"}))
            st.caption("Si el p-value es menor a 0.05, se rechaza la hipótesis de normalidad.")

            import seaborn as sns
            from scipy.stats import norm

                          # ----- Histograma con curva normal -----
            st.subheader("Distribución de retornos vs Normal")

                          # Selector de activo
            ticker_sel = st.selectbox("Elegí un activo para graficar", log_returns.columns)

                          # Datos del activo
            data = log_returns[ticker_sel]

                          # Histograma + curva normal ajustada
            fig, ax = plt.subplots()
            sns.histplot(data, kde=False, stat="density", bins=30, ax=ax, color="skyblue")

                          # Ajuste de normal con misma media y desvío
            mu, sigma = data.mean(), data.std()
            x = np.linspace(data.min(), data.max(), 100)
            ax.plot(x, norm.pdf(x, mu, sigma), 'r', linewidth=2)

            ax.set_title(f"Histograma de {ticker_sel} con curva normal")
            st.pyplot(fig)


