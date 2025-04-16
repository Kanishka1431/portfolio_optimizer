from django import forms

class PortfolioForm(forms.Form):
    tickers = forms.CharField(label='Stock Tickers (comma separated)', max_length=200)
    risk_free_rate = forms.FloatField(label='Risk-Free Rate (%)', min_value=0)
    use_real_data = forms.BooleanField(
        label='Use Real-Time Stock Data',
        required=False,
        initial=False
    )
    
    def clean_risk_free_rate(self):
        rate = self.cleaned_data['risk_free_rate']
        return rate / 100
