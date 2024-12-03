import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize
import datetime
import streamlit as st
import matplotlib.dates as mdates
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])

with tab1:
   # Example: Reading the data
   df = pd.read_csv('cumulative_returns.csv', index_col='date')
   vw_returns = pd.read_csv('vw_cumreturns.csv', index_col='date')
   sp500_returns = pd.read_csv('sp500data.csv', index_col='Date')
   sp500_returns = sp500_returns['Daily_Returns']
   sp500_returns = (1 + sp500_returns).cumprod()
   vw_returns.index = pd.to_datetime(vw_returns.index, errors='coerce')
   sp500_returns.index = pd.to_datetime(sp500_returns.index, errors='coerce')
   # Streamlit UI for parameters
   options_tc = [0.0, 0.01]
   options_constraint_weight = ['constraint_no_min', 'constraint_min_1percent', 'constraint_min_2percent', 'constraint_min_3percent', 'constraint_short_sell_ok']

   phi_tab1 = st.slider("Choose a phi value", min_value=1, max_value=10, step=1, key='phi_tab1')
   omega_tab1 = st.slider("Choose an omega", min_value=0.0, max_value=0.5, step=0.1, key='omega_tab1')
   tc = st.selectbox("Transaction cost", options_tc)
   constraint_weight = st.selectbox("Choose a constraint", options_constraint_weight)

   # Ensure the "Value Weighted" column matches the selected transaction cost
   selected_tc_column = f"{tc}"  # Use the selected transaction cost
   if selected_tc_column in vw_returns.columns:
      vw_returns_selected = vw_returns[selected_tc_column]
   else:
      st.error(f"Transaction cost column '{selected_tc_column}' not found in vw_returns.")        vw_returns_selected = vw_returns.iloc[:, 0]  # Default to the first column
      # Convert indices to datetime if not already done
      vw_returns_selected.index = pd.to_datetime(vw_returns_selected.index, errors='coerce')
      vw_returns_selected = vw_returns_selected.dropna()


      # Dynamically ensure that the Current Series and VW Returns have the same transaction cost
      current_serie_column = f"phi_{phi_tab1}_tc_{tc}_omega_{omega_tab1}_{constraint_weight}"
   if current_serie_column in df.columns:
      current_serie = df[current_serie_column]
   else:
      st.error(f"Column '{current_serie_column}' not found in current series data.")
      current_serie = df.iloc[:, 0]  # Default to the first column

   # Select the current series based on user input
   current_serie = df[f"phi_{phi_tab1}_tc_{tc}_omega_{omega_tab1}_{constraint_weight}"]
   current_serie.index = pd.to_datetime(current_serie.index, errors='coerce')
   current_serie = current_serie.dropna()
   start_date = datetime.datetime(2018, 1, 1)  # Fixed start date
   end_date = datetime.datetime(2020, 3, 31)  # Fixed end date

   if st.button("Generate Graph and Metrics"):
      fig, ax = plt.subplots(figsize=(10, 6))

      # Plot Current Series and Selected VW Returns
      ax.plot(current_serie.index, current_serie, color='blue', label="Black-Litterman")
      ax.plot(vw_returns_selected.index, vw_returns_selected, color='green', label=f"Value Weighted (tc={tc})")
      ax.plot(sp500_returns.index, sp500_returns, color='red', label="S&P500")

      # Title and axis labels
      ax.set_title("Cumulative Returns Over Time")
      ax.set_xlabel("Date")
      ax.set_ylabel("Cumulative Returns")

      # X and Y limits
      ax.set_xlim(start_date, end_date)
      y_min = min(current_serie.min().min(), vw_returns_selected.min().min(), sp500_returns.min().min()) * 0.95
      y_max = max(current_serie.max().max(), vw_returns_selected.max().max(), sp500_returns.max().max()) * 1.05
      ax.set_ylim(y_min, y_max)

      # Legend positioning
      ax.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place legend outside the plot

      # Format x-axis for dates
      ax.xaxis.set_major_locator(mdates.YearLocator())
      ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
      ax.tick_params(axis='x', rotation=45)

      #Force Streamlit to render the plot correctly
      st.pyplot(fig, clear_figure=True)


      # Calculate metrics
      daily_returns = current_serie.pct_change().dropna()

      # Geometric mean daily return
      geometric_mean_daily_return = (np.prod(1 + daily_returns) ** (1 / len(daily_returns))) - 1

      # Standard deviation of daily returns
      std_daily_return = daily_returns.std()

      # Risk-free rate
      daily_risk_free_rate = 0.000055

      # Sharpe ratio and annualized metrics
      sharpe_ratio = (geometric_mean_daily_return - daily_risk_free_rate) / std_daily_return
      annualized_sharpe_ratio = sharpe_ratio * np.sqrt(252)

      # Annualized return
      cumulative_return_end = current_serie.iloc[-1]
      trading_days = len(current_serie)
      years = trading_days / 252
      annualized_return = (cumulative_return_end) ** (1 / years) - 1

      # Annualized volatility
      annualized_std = std_daily_return * np.sqrt(252)

      # Display metrics
      st.markdown("### Portfolio Performance Metrics")
      col1, col2, col3 = st.columns(3)
      col1.metric("Annualized Return", f"{annualized_return:.2%}")
      col2.metric("Annualized Volatility", f"{annualized_std:.2%}")
      col3.metric("Annualized Sharpe Ratio", f"{annualized_sharpe_ratio:.2f}")

with tab2:

    phi_tab2 = st.slider("Choose a phi value", min_value=1, max_value=10, step=1, key='phi_tab2')
    omega_tab2 = st.slider("Choose an omega value", min_value=0.1, max_value=0.5, step=0.1, key='omega_tabb2')
    options_weight_constraint2 = [0, 0.01, 0.02, 0.03]
    constraint_weight2 = st.selectbox("Choose a minimum weight", options_weight_constraint2, key='constraint_weight2')



    def load_dataframes():
        file_paths = {
        "beta_x_pola": 'beta_x_var_pola.csv',
        "df_polarity": 'df_polarity.csv',
        "df_returns": 'df_returns.csv',
        "rf_futures": 'rf_futures.csv',
        "x0_futures": 'x0_futures.csv',
        "vcv": 'vcv_matrix.csv',
    }
        dataframes = {}
        for key, path in file_paths.items():
            if key == "vcv":
                dataframes[key] = pd.read_csv(path)  # vcv doesn't use index_col='Date'
            else:
                dataframes[key] = pd.read_csv(path, index_col='Date')  # Set Date as index
        return dataframes


    tickers = ['AAPL', 'AMD', 'AMRN', 'AMZN', 'BABA', 'BAC', 'BB', 'FB', 'GLD',
            'IWM', 'JNUG', 'MNKD', 'NFLX', 'PLUG', 'QQQ', 'SPY', 'TSLA', 'UVXY']

    def matrix_P(tickers, selected_tickers):
        n = len(tickers)
        # Initialize a zero matrix
        P = np.zeros((n, n))

        # Set diagonal elements to 1 for selected tickers
        for i, ticker in enumerate(tickers):
            if ticker in selected_tickers:
                P[i, i] = 1

        # Identify rows with all zeros
        non_zero_rows = ~np.all(P == 0, axis=1)  # Identify rows that are not all zeros
        retained_indices = np.where(non_zero_rows)[0]  # Indices of rows that are retained

        # Keep only rows that are not all zeros
        P = P[non_zero_rows]  # Retain rows with non-zero values

        return P, retained_indices
    
    
    # Track selected tickers
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = []

    columns = st.columns(3)
    for idx, ticker in enumerate(tickers):
        col_idx = idx % 3
        with columns[col_idx]:
            if st.button(f"{ticker}", key=ticker):
                if ticker not in st.session_state.selected_tickers:
                    st.session_state.selected_tickers.append(ticker)

    if st.session_state.selected_tickers:
        st.markdown("### Selected Tickers:")
        st.write(", ".join([f"**{ticker}**" for ticker in st.session_state.selected_tickers]))
    else:
        st.markdown("### Selected Tickers:")
        st.write("No tickers selected.")
            # Add a "Select All" button
    if st.button("Select All"):
        st.session_state.selected_tickers = tickers
    if st.button("Reset Selection"):
        st.session_state.selected_tickers = []
    # Generate P matrix if tickers are selected
    if st.session_state.selected_tickers:
        P, retained_indices = matrix_P(tickers, st.session_state.selected_tickers)

    else:
        P, retained_indices = matrix_P(tickers, st.session_state.selected_tickers)
        st.write(P)

    
    def QP(x, sigma, mu, phi ):
        gamma = 1/phi
        
        v = 0.5 * x.T @ sigma @ x - gamma * x.T @ mu
        
        return v

    def gamma_matrix(tau,vcv):
        gamma_matrix = tau * vcv
        
        return gamma_matrix
    def calculate_implied_mu(vcv, x0, rf, phi):
        gamma = 1/phi
        ones = np.ones(18)
        r_vector = rf.values.item() * ones
        mu = r_vector.reshape(-1, 1) + (1/gamma) * vcv @ x0.values.reshape(-1, 1)
        mu = mu.flatten() 
        return mu



    def calculate_optimized_weights(vcv,x0,rf,P,Q,retained_indices, omega,phi,tc,df_returns,constraints):
        omega_matrix = np.eye(P.shape[0]) * omega
        mu_tilda = calculate_implied_mu(vcv, x0, rf, phi)
        retained_indices = retained_indices
        Q = Q[retained_indices]
        mu_bar = mu_tilda + (gamma_matrix(tau,vcv) @ P.T) @ np.linalg.inv(P @ gamma_matrix(tau,vcv) @ P.T + omega_matrix) @ (Q - P @ mu_tilda)
        constraints = constraints
        res = minimize(QP, x0, args = (vcv / np.max(np.abs(vcv)), mu_bar / np.max(np.abs(mu_bar)), phi) , options={'disp': False}, constraints = constraints)

        optimized_weights = res.x
        optimized_weights = pd.DataFrame(optimized_weights)
        optimized_weights.index = x0.index
        ptf_return = (df_returns.iloc[-2] * optimized_weights.values.flatten()).sum()
        optimized_weights = optimized_weights.iloc[:, 0]
        optimized_weights.name = x0.name

        return optimized_weights,ptf_return


    # Load all DataFrames
    dataframes = load_dataframes()

    # Access specific DataFrames
    beta_x_pola = dataframes["beta_x_pola"]
    df_polarity = dataframes["df_polarity"]
    df_returns = dataframes["df_returns"]
    rf_futures = dataframes["rf_futures"]
    x0_futures = dataframes["x0_futures"]
    vcv_df = dataframes["vcv"]
    vcv = vcv_df.to_numpy()
    tau=0.05
    tc=0
    rf = 0.000008
    Q= calculate_implied_mu(vcv,x0_futures.iloc[-2],rf_futures.iloc[-3],phi_tab2)
    P, retained_indices= matrix_P(tickers, st.session_state.selected_tickers)   
    constraints = [
        LinearConstraint(np.ones(x0_futures.iloc[0].shape), ub=1), 
        LinearConstraint(-np.ones(x0_futures.iloc[0].shape), ub=-1),
        LinearConstraint(np.eye(x0_futures.iloc[0].shape[0]), lb=constraint_weight2)
    ]

        
        
    if st.button("Run Optimization"):

        if not st.session_state.selected_tickers:
            st.error("Please select at least 1 ticker before running the optimization.")

        else : 

            opti_weight= calculate_optimized_weights(vcv,x0_futures.iloc[-2],rf_futures.iloc[-3],
                                                     P,Q,retained_indices,omega_tab2,phi_tab2,tc,
                                                     df_returns.iloc[-2],constraints=constraints)[0]

            #compute returns BL pf
            weights_opti = opti_weight
            return_20 = df_returns.iloc[-2]
            return_pf_opti = (weights_opti * return_20).sum()

            optimized_weights = opti_weight
            # Convert weights to percentages
            optimized_weights_percent = optimized_weights * 100
            weights_series = pd.Series(optimized_weights_percent, index=tickers).round(4)

            threshold = 3  # Increase the threshold for grouping
            small_weights = weights_series[weights_series < threshold].sum()
            weights_series = weights_series[weights_series >= threshold]
            if small_weights > 0:
                weights_series['Other'] = small_weights

            # Sort by weight for a cleaner legend
            weights_series = weights_series.sort_values(ascending=False)
            # Convert Series to DataFrame with proper column names
            weights_df = weights_series.reset_index()
            weights_df.columns = ["Asset", "Weight (%)"]  # Rename columns

            # Display the table in Streamlit
            st.write("### Optimized Portfolio Weights (in %):")
            st.table(weights_df.style.format({"Weight (%)": "{:.2f}"}))

            fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figure size
            ax.pie(
                weights_series, 
                labels=None,  # Remove labels from the chart
                startangle=90, 
                autopct=lambda p: f'{p:.0f}%' if p >= threshold else ''  # Show percentages only for larger slices
            )
            ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
            plt.title("Portfolio Weights Distribution")

            # Add a legend for the slices
            plt.legend(
                labels=weights_series.index,
                loc='best',
                title="Assets",
                bbox_to_anchor=(1, 0.5),  # Move legend to the side
            )
            
            st.pyplot(fig) 
            

            value_weights = x0_futures.iloc[-2].values

            # Get the returns for the last two days
            #second_to_last_day = df_returns.iloc[-2]
            last_day = df_returns.iloc[-2]

            # Compute the portfolio return for the value-weighted portfolio
            portfolio_return = (value_weights * last_day).sum()
            VW_portfolio_return = portfolio_return

            last_index = df_returns.index[-2]

            if 'SPY' in df_returns.columns:  # Ensure SPY exists in the dataset
                sp500_return = df_returns.loc[last_index, 'SPY']
                sp500_return = sp500_return
            else:
                sp500_return = np.nan  # Fallback if SPY column is missing


    
            # Display performance comparison
            st.write("Performance Comparison (Next Day):")
            performance_comparison = pd.DataFrame({
                "Portfolio": [return_pf_opti],
                "Value-Weighted Portfolio": [VW_portfolio_return],
                "S&P 500": [sp500_return]
            }).T
            
            performance_comparison.columns = ["Next Day Return (%)"]
            performance_comparison["Next Day Return (%)"]*=100 # Convert to %
            st.table(performance_comparison)
            
            # Create a bar chart for comparison
            fig, ax = plt.subplots()
            performance_comparison["Next Day Return (%)"].plot(kind="bar", ax=ax, color=["blue", "green", "red"])
            ax.set_title("Next Day Return Comparison")
            ax.set_ylabel("Return (%)")
            ax.grid(axis="y")
            st.pyplot(fig)






