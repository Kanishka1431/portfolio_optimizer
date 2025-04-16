from django.shortcuts import render
from .forms import PortfolioForm
import pandas as pd
import numpy as np
import datetime
from scipy.optimize import minimize
import json
from django.http import HttpResponse
from django.template.loader import get_template
from xhtml2pdf import pisa
import yfinance as yf

# Simulated data
def get_simulated_data(tickers):
    np.random.seed(42)
    dates = pd.date_range(end=datetime.datetime.today(), periods=252)
    data = pd.DataFrame(index=dates)

    for ticker in tickers:
        prices = np.cumprod(1 + np.random.normal(0.0005, 0.02, size=len(dates))) * 100
        data[ticker] = prices

    return data

# Real-time data using yfinance
import yfinance as yf
import pandas as pd

def get_real_data(tickers):
    import yfinance as yf
    import pandas as pd

    data = pd.DataFrame()

    for ticker in tickers:
        try:
            df = yf.download(ticker, period='1y', auto_adjust=True)
            # Use 'Close' if auto_adjust is True, equivalent to 'Adj Close'
            if not df.empty and 'Close' in df.columns:
                data[ticker] = df['Close']
            else:
                print(f"No 'Close' data for {ticker}")
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    return data


def get_risk_level(vol):
    if vol < 20:
        return "Low"
    elif vol < 30:
        return "Medium"
    else:
        return "High"

def compute_drawdown(series):
    cumulative = (1 + series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown


def optimize_portfolio(data, risk_free_rate):
    daily_returns = data.pct_change()
    returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252
    individual_volatility = np.sqrt(np.diag(cov_matrix))
    num_assets = len(returns)

    def portfolio_perf(weights):
        ret = np.dot(weights, returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (ret - risk_free_rate) / vol
        return -sharpe

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets]

    result = minimize(portfolio_perf, init_guess, method='SLSQP',
                      bounds=bounds, constraints=[constraints])

    optimal_weights = result.x
    expected_return = np.dot(optimal_weights, returns)
    volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    sharpe_ratio = (expected_return - risk_free_rate) / volatility

    # Generate basic insights per stock
    insights = {}
    for i, (stock, weight) in enumerate(zip(returns.index, optimal_weights)):
        reason = []
        if returns[stock] > expected_return:
            reason.append("has shown strong returns recently")
        if individual_volatility[i] < volatility:
            reason.append("has lower risk than the overall portfolio")
        if weight > 0.2:
            reason.append("was heavily weighted due to favorable risk-return profile")

        insight = f"{stock} was assigned {weight*100:.2f}% weight because it " + ", ".join(reason) + "."
        insights[stock] = insight

    return {
        'weights': {
            stock: {
                'weight': f"{weight * 100:.2f}%",
                'risk': f"{individual_volatility[i] * 100:.2f}%",
                'risk_level': get_risk_level(individual_volatility[i] * 100),
                'insight': insights[stock]
            }
            for i, (stock, weight) in enumerate(zip(returns.index, optimal_weights))
        },
        'expected_return': f"{expected_return * 100:.2f}%",
        'volatility': f"{volatility * 100:.2f}%",
        'sharpe_ratio': f"{sharpe_ratio:.2f}"
    }

def home(request):
    if request.method == 'POST':
        form = PortfolioForm(request.POST)
        if form.is_valid():
            tickers = [ticker.strip().upper() for ticker in form.cleaned_data['tickers'].split(',')]
            risk_free = form.cleaned_data['risk_free_rate'] / 100  # Convert % to decimal
            use_real_data = form.cleaned_data['use_real_data']

            data = get_real_data(tickers) if use_real_data else get_simulated_data(tickers)

            # ✅ Check for any missing tickers in real-time data
            missing = list(set(tickers) - set(data.columns))
            if data.empty or len(data.columns) < len(tickers):
                error_message = 'Failed to fetch stock data.'
                if missing:
                    error_message += f" Could not retrieve data for: {', '.join(missing)}"
                return render(request, 'home.html', {
                    'form': form,
                    'error': error_message
                })

            result = optimize_portfolio(data, risk_free)

            price_data = {
                ticker: {
                    date.strftime('%Y-%m-%d'): round(price, 2)
                    for date, price in data[ticker].items()
                }
                for ticker in data.columns
            }

            request.session['result'] = result
            request.session['tickers'] = tickers

            return render(request, 'results.html', {
                'result': result,
                'price_data': json.dumps(price_data)
            })
    else:
        form = PortfolioForm()

    return render(request, 'home.html', {'form': form})


def generate_pdf(request):
    if request.method == 'POST':
        result = request.session.get('result')
        tickers = request.session.get('tickers')
        chart_image = request.POST.get('chart_image')

        if not result or not chart_image:
            return HttpResponse("Missing data for PDF", status=400)

        template = get_template("portfolio_pdf.html")
        html = template.render({
            'result': result,
            'tickers': tickers,
            'chart_image': chart_image
        })

        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="portfolio_report.pdf"'

        pisa_status = pisa.CreatePDF(html, dest=response)
        if pisa_status.err:
            return HttpResponse('PDF generation error', status=500)

        return response
    return HttpResponse("Invalid method", status=405)
