"""Export dashboard-style plots from the saved, lightweight checkpoint inventory."""
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUT = Path(__file__).parent

def main():
    records = {r['name']: r for r in json.loads((OUT/'checkpoint-inventory.json').read_text())}
    runs = [
        ('model-32mil-param-quick-tf-decay.latest.pth', 'Six layers: scheduled TF; historical AR validation'),
        ('model-expanded.latest.pth', 'Ten layers: TF 0.20 → 0; historical AR validation'),
        ('model-100mil.latest.pth', 'Six layers, noisy-v1: TF 1; TF validation'),
    ]
    titles = [f'{title}<br>{metric}' for _, title in runs
              for metric in ['Train and validation loss', 'Validation − train loss', 'Teacher-forcing ratio']]
    fig = make_subplots(rows=3, cols=3, subplot_titles=titles, vertical_spacing=0.12, horizontal_spacing=0.08)
    for row, (name, title) in enumerate(runs, 1):
        history = records[name]['history']
        x = [h['epoch'] for h in history]
        train = [h['train_loss'] for h in history]
        val = [h['validation_loss'] for h in history]
        tf = [h['teacher_forcing_ratio'] for h in history]
        for label, y, color in [('Train', train, '#2563eb'), ('Validation', val, '#ea580c')]:
            fig.add_trace(go.Scatter(x=x,y=y,name=label,legendgroup=label,showlegend=row==1,
                                    mode='lines+markers',line=dict(color=color),customdata=tf,
                                    hovertemplate='Interval %{x}<br>Loss %{y:.4f}<br>TF %{customdata:.3f}<extra>%{fullData.name}</extra>'),row=row,col=1)
        fig.add_trace(go.Scatter(x=x,y=[v-t for v,t in zip(val,train)],name='Validation − train',
                                legendgroup='gap',showlegend=row==1,mode='lines+markers',line=dict(color='#dc2626')),row=row,col=2)
        fig.add_hline(y=0,line_color='#94a3b8',line_dash='dot',row=row,col=2)
        fig.add_trace(go.Scatter(x=x,y=tf,name='Teacher forcing',legendgroup='tf',showlegend=row==1,
                                mode='lines+markers',line=dict(color='#7c3aed')),row=row,col=3)
        fig.update_yaxes(range=[-0.03,1.03],row=row,col=3)
        for col in [1,2,3]: fig.update_xaxes(title_text='Logged validation interval (not wall time)',row=row,col=col)
    fig.update_annotations(font_size=12)
    fig.update_layout(template='plotly_white',height=1250,width=1500,
                      title='September 2026 training histories — loss scales differ between experiments',
                      legend=dict(orientation='h',y=1.065),margin=dict(t=140,b=90),hovermode='x unified')
    (OUT/'assets').mkdir(exist_ok=True)
    fig.write_html(OUT/'assets/training-dashboard.html',include_plotlyjs=True,full_html=True)
    print('Saved self-contained dashboard (works offline).')

if __name__ == '__main__': main()
