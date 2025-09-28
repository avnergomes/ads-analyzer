"""
Aplicativo Streamlit atualizado para trabalhar com planilha pública
Mapeamento detalhado e integração com dados de anúncios
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import io
import warnings
from public_sheets_connector import PublicSheetsConnector

warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Ads Analyzer v2.0 - Integração Completa",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin: 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .data-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class AdsDataProcessor:
    """Processador de dados de anúncios com mapeamento inteligente"""
    
    @staticmethod
    def detect_and_normalize_columns(df):
        """
        Detecta e normaliza colunas de dados de anúncios
        """
        if df is None or df.empty:
            return None
        
        # Mapeamento flexível para diferentes formatos de arquivo
        column_mappings = {
            'date': [
                'date', 'day', 'date_start', 'reporting_starts', 'data', 'fecha',
                'datum', 'created_time', 'campaign_date'
            ],
            'campaign_name': [
                'campaign_name', 'campaign', 'campaign_id', 'campaign_title',
                'nombre_campana', 'campagne', 'kampagne'
            ],
            'impressions': [
                'impressions', 'impression', 'impr', 'views', 'reach',
                'impresiones', 'impressionen', 'show'
            ],
            'clicks': [
                'clicks', 'click', 'link_clicks', 'website_clicks',
                'clics', 'klicks', 'tap'
            ],
            'spend': [
                'spend', 'cost', 'amount_spent', 'investment', 'budget',
                'gasto', 'kosten', 'costo', 'investimento'
            ],
            'conversions': [
                'conversions', 'conversion', 'results', 'actions', 'leads',
                'conversiones', 'konversionen', 'resultados'
            ],
            'ctr': [
                'ctr', 'click_through_rate', 'clickthrough_rate',
                'tasa_clics', 'klickrate'
            ],
            'cpc': [
                'cpc', 'cost_per_click', 'avg_cpc', 'costo_por_clic',
                'kosten_pro_klick'
            ],
            'cpa': [
                'cpa', 'cost_per_acquisition', 'cost_per_action',
                'costo_por_adquisicion', 'kosten_pro_akquisition'
            ],
            'cpm': [
                'cpm', 'cost_per_mille', 'cost_per_1000_impressions',
                'costo_por_mil', 'kosten_pro_tausend'
            ]
        }
        
        # Aplicar mapeamento
        normalized_df = df.copy()
        
        for standard_col, variations in column_mappings.items():
            for col in normalized_df.columns:
                col_clean = col.lower().replace(' ', '_').replace('-', '_')
                if col_clean in [v.lower() for v in variations]:
                    if standard_col not in normalized_df.columns:
                        normalized_df = normalized_df.rename(columns={col: standard_col})
                        break
        
        return normalized_df
    
    @staticmethod
    def calculate_missing_kpis(df):
        """Calcula KPIs que possam estar faltando"""
        if df is None or df.empty:
            return df
        
        # CTR
        if 'ctr' not in df.columns and 'impressions' in df.columns and 'clicks' in df.columns:
            df['ctr'] = np.where(df['impressions'] > 0, 
                               (df['clicks'] / df['impressions']) * 100, 0)
        
        # CPC
        if 'cpc' not in df.columns and 'spend' in df.columns and 'clicks' in df.columns:
            df['cpc'] = np.where(df['clicks'] > 0, 
                               df['spend'] / df['clicks'], 0)
        
        # CPA
        if 'cpa' not in df.columns and 'spend' in df.columns and 'conversions' in df.columns:
            df['cpa'] = np.where(df['conversions'] > 0, 
                               df['spend'] / df['conversions'], 0)
        
        # CPM
        if 'cpm' not in df.columns and 'spend' in df.columns and 'impressions' in df.columns:
            df['cpm'] = np.where(df['impressions'] > 0,
                               (df['spend'] / df['impressions']) * 1000, 0)
        
        # Conversion Rate
        if 'conversion_rate' not in df.columns and 'clicks' in df.columns and 'conversions' in df.columns:
            df['conversion_rate'] = np.where(df['clicks'] > 0,
                                           (df['conversions'] / df['clicks']) * 100, 0)
        
        return df
    
    @staticmethod
    def prepare_for_integration(ads_df, sales_df):
        """
        Prepara dados para integração entre anúncios e vendas
        """
        if ads_df is None or sales_df is None:
            return ads_df, sales_df, {}
        
        integration_mapping = {}
        
        # Criar chaves de integração baseadas em datas
        if 'date' in ads_df.columns and 'show_date' in sales_df.columns:
            # Converter datas para formato comum
            ads_df['integration_date'] = pd.to_datetime(ads_df['date']).dt.date
            sales_df['integration_date'] = pd.to_datetime(sales_df['show_date']).dt.date
            
            integration_mapping['date_range'] = {
                'ads_start': ads_df['integration_date'].min(),
                'ads_end': ads_df['integration_date'].max(),
                'sales_start': sales_df['integration_date'].min(),
                'sales_end': sales_df['integration_date'].max()
            }
        
        # Mapear campanhas para cidades se possível
        if 'campaign_name' in ads_df.columns and 'city' in sales_df.columns:
            # Tentar extrair cidade do nome da campanha
            ads_df['extracted_city'] = ads_df['campaign_name'].str.extract(r'([A-Za-z\s]+)', expand=False)
            
            unique_cities = sales_df['city'].dropna().unique()
            unique_campaigns = ads_df['campaign_name'].dropna().unique()
            
            integration_mapping['city_campaign_mapping'] = {
                'cities_in_sales': unique_cities.tolist(),
                'campaigns_in_ads': unique_campaigns.tolist()
            }
        
        return ads_df, sales_df, integration_mapping

class IntegratedDashboard:
    """Dashboard integrado para análise completa"""
    
    def __init__(self):
        self.sales_data = None
        self.ads_data = None
        self.integration_mapping = {}
    
    def create_sales_overview(self, df):
        """Cria visão geral dos dados de vendas"""
        if df is None or df.empty:
            st.warning("📊 Nenhum dado de vendas disponível")
            return
        
        st.subheader("🎫 Visão Geral das Vendas")
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_shows = len(df)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Total de Shows</div>
                <div class="metric-value">{total_shows:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_capacity = df['capacity'].sum() if 'capacity' in df.columns else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Capacidade Total</div>
                <div class="metric-value">{total_capacity:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_sold = df['total_sold'].sum() if 'total_sold' in df.columns else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Total Vendido</div>
                <div class="metric-value">{total_sold:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            total_revenue = df['sales_to_date'].sum() if 'sales_to_date' in df.columns else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Receita Total</div>
                <div class="metric-value">R$ {total_revenue:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Métricas secundárias
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            avg_occupancy = df['occupancy_rate'].mean() if 'occupancy_rate' in df.columns else 0
            color_class = "status-success" if avg_occupancy >= 80 else "status-warning" if avg_occupancy >= 60 else "status-error"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Ocupação Média</div>
                <div class="metric-value {color_class}">{avg_occupancy:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            avg_ticket = df['avg_ticket_price'].mean() if 'avg_ticket_price' in df.columns else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Ticket Médio</div>
                <div class="metric-value">R$ {avg_ticket:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col7:
            cities_count = df['city'].nunique() if 'city' in df.columns else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Cidades</div>
                <div class="metric-value">{cities_count}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col8:
            sold_out_shows = len(df[df['occupancy_rate'] >= 99]) if 'occupancy_rate' in df.columns else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Shows Esgotados</div>
                <div class="metric-value">{sold_out_shows}</div>
            </div>
            """, unsafe_allow_html=True)
    
    def create_sales_charts(self, df):
        """Cria gráficos de vendas"""
        if df is None or df.empty:
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Performance por Cidade**")
            if 'city' in df.columns and 'total_sold' in df.columns:
                city_performance = df.groupby('city').agg({
                    'total_sold': 'sum',
                    'capacity': 'sum',
                    'sales_to_date': 'sum'
                }).reset_index()
                
                city_performance['occupancy'] = (city_performance['total_sold'] / city_performance['capacity']) * 100
                city_performance = city_performance.sort_values('total_sold', ascending=True)
                
                fig = px.bar(city_performance.tail(10), 
                           x='total_sold', y='city', 
                           orientation='h',
                           color='occupancy',
                           color_continuous_scale='RdYlGn',
                           title="Top 10 Cidades por Vendas")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**🎯 Distribuição de Ocupação**")
            if 'occupancy_rate' in df.columns:
                fig = px.histogram(df, x='occupancy_rate', 
                                 nbins=20,
                                 title="Distribuição de Taxa de Ocupação",
                                 color_discrete_sequence=['#1f77b4'])
                fig.update_layout(height=400)
                fig.update_xaxis(title="Taxa de Ocupação (%)")
                fig.update_yaxis(title="Número de Shows")
                st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico de linha temporal
        if 'show_date' in df.columns and df['show_date'].notna().any():
            st.markdown("**📅 Vendas ao Longo do Tempo**")
            
            daily_sales = df.groupby('show_date').agg({
                'today_sold': 'sum',
                'sales_to_date': 'sum',
                'total_sold': 'sum'
            }).reset_index()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=daily_sales['show_date'],
                y=daily_sales['total_sold'],
                mode='lines+markers',
                name='Total Vendido',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=daily_sales['show_date'],
                y=daily_sales['today_sold'],
                mode='lines+markers',
                name='Vendas do Dia',
                line=dict(color='#ff7f0e', width=2)
            ))
            
            fig.update_layout(
                title="Evolução das Vendas por Data",
                xaxis_title="Data do Show",
                yaxis_title="Quantidade de Ingressos",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def create_ads_overview(self, df):
        """Cria visão geral dos dados de anúncios"""
        if df is None or df.empty:
            st.warning("📊 Nenhum dado de anúncios disponível")
            return
        
        st.subheader("📈 Visão Geral dos Anúncios")
        
        # Métricas principais de anúncios
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_impressions = df['impressions'].sum() if 'impressions' in df.columns else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Impressões</div>
                <div class="metric-value">{total_impressions:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_clicks = df['clicks'].sum() if 'clicks' in df.columns else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Clicks</div>
                <div class="metric-value">{total_clicks:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_spend = df['spend'].sum() if 'spend' in df.columns else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Investimento</div>
                <div class="metric-value">R$ {total_spend:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            total_conversions = df['conversions'].sum() if 'conversions' in df.columns else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Conversões</div>
                <div class="metric-value">{total_conversions:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # KPIs calculados
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            avg_ctr = df['ctr'].mean() if 'ctr' in df.columns else 0
            color_class = "status-success" if avg_ctr >= 2 else "status-warning" if avg_ctr >= 1 else "status-error"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">CTR Médio</div>
                <div class="metric-value {color_class}">{avg_ctr:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            avg_cpc = df['cpc'].mean() if 'cpc' in df.columns else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">CPC Médio</div>
                <div class="metric-value">R$ {avg_cpc:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col7:
            avg_cpa = df['cpa'].mean() if 'cpa' in df.columns else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">CPA Médio</div>
                <div class="metric-value">R$ {avg_cpa:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col8:
            conversion_rate = df['conversion_rate'].mean() if 'conversion_rate' in df.columns else 0
            color_class = "status-success" if conversion_rate >= 5 else "status-warning" if conversion_rate >= 2 else "status-error"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Taxa Conversão</div>
                <div class="metric-value {color_class}">{conversion_rate:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
    def create_ads_charts(self, df):
        """Cria gráficos de anúncios"""
        if df is None or df.empty:
            return
        
        # Funil de conversão
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Funil de Conversão**")
            
            funnel_data = []
            if 'impressions' in df.columns:
                funnel_data.append(['Impressões', df['impressions'].sum()])
            if 'clicks' in df.columns:
                funnel_data.append(['Clicks', df['clicks'].sum()])
            if 'conversions' in df.columns:
                funnel_data.append(['Conversões', df['conversions'].sum()])
            
            if len(funnel_data) >= 2:
                labels, values = zip(*funnel_data)
                
                fig = go.Figure(go.Funnel(
                    y=labels,
                    x=values,
                    textinfo="value+percent initial+percent previous",
                    marker=dict(color=["#1f77b4", "#ff7f0e", "#2ca02c"][:len(labels)])
                ))
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**📊 Performance por Campanha**")
            
            if 'campaign_name' in df.columns:
                campaign_performance = df.groupby('campaign_name').agg({
                    'impressions': 'sum',
                    'clicks': 'sum',
                    'spend': 'sum',
                    'conversions': 'sum'
                }).reset_index()
                
                campaign_performance['ctr'] = np.where(
                    campaign_performance['impressions'] > 0,
                    (campaign_performance['clicks'] / campaign_performance['impressions']) * 100,
                    0
                )
                
                fig = px.scatter(campaign_performance,
                               x='spend', y='conversions',
                               size='clicks',
                               color='ctr',
                               hover_data=['campaign_name'],
                               title="Investimento vs Conversões por Campanha",
                               color_continuous_scale='RdYlGn')
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Tendência temporal
        if 'date' in df.columns and df['date'].notna().any():
            st.markdown("**📈 Tendência Temporal**")
            
            daily_ads = df.groupby('date').agg({
                'impressions': 'sum',
                'clicks': 'sum',
                'spend': 'sum',
                'conversions': 'sum'
            }).reset_index()
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Impressões', 'Clicks', 'Investimento', 'Conversões']
            )
            
            fig.add_trace(
                go.Scatter(x=daily_ads['date'], y=daily_ads['impressions'],
                         name='Impressões', line=dict(color='#1f77b4')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=daily_ads['date'], y=daily_ads['clicks'],
                         name='Clicks', line=dict(color='#ff7f0e')),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(x=daily_ads['date'], y=daily_ads['spend'],
                         name='Investimento', line=dict(color='#2ca02c')),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=daily_ads['date'], y=daily_ads['conversions'],
                         name='Conversões', line=dict(color='#d62728')),
                row=2, col=2
            )
            
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    def create_integration_analysis(self, sales_df, ads_df, mapping):
        """Cria análise integrada entre vendas e anúncios"""
        if sales_df is None or ads_df is None:
            return
        
        st.subheader("🔗 Análise Integrada")
        
        # Verificar sobreposição de datas
        if 'integration_date' in sales_df.columns and 'integration_date' in ads_df.columns:
            
            sales_dates = set(sales_df['integration_date'].dropna())
            ads_dates = set(ads_df['integration_date'].dropna())
            overlap_dates = sales_dates.intersection(ads_dates)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Datas com Vendas</div>
                    <div class="metric-value">{len(sales_dates)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Datas com Anúncios</div>
                    <div class="metric-value">{len(ads_dates)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                overlap_pct = (len(overlap_dates) / max(len(sales_dates), 1)) * 100
                color_class = "status-success" if overlap_pct >= 70 else "status-warning" if overlap_pct >= 40 else "status-error"
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Sobreposição</div>
                    <div class="metric-value {color_class}">{overlap_pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Análise de correlação por data
            if len(overlap_dates) > 0:
                # Agregar dados por data
                sales_by_date = sales_df.groupby('integration_date').agg({
                    'total_sold': 'sum',
                    'sales_to_date': 'sum'
                }).reset_index()
                
                ads_by_date = ads_df.groupby('integration_date').agg({
                    'impressions': 'sum',
                    'clicks': 'sum',
                    'spend': 'sum',
                    'conversions': 'sum'
                }).reset_index()
                
                # Fazer merge
                integrated_data = pd.merge(
                    sales_by_date, ads_by_date, 
                    on='integration_date', 
                    how='inner'
                )
                
                if not integrated_data.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📊 Investimento vs Vendas**")
                        fig = px.scatter(
                            integrated_data,
                            x='spend', y='total_sold',
                            size='impressions',
                            color='clicks',
                            hover_data=['integration_date'],
                            title="Correlação: Investimento em Anúncios vs Vendas",
                            trendline="ols"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("**🎯 Impressões vs Receita**")
                        fig = px.scatter(
                            integrated_data,
                            x='impressions', y='sales_to_date',
                            size='clicks',
                            color='spend',
                            hover_data=['integration_date'],
                            title="Correlação: Impressões vs Receita",
                            trendline="ols"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Calcular correlações
                    st.markdown("**📈 Correlações**")
                    
                    correlations = {
                        'Investimento vs Vendas': integrated_data['spend'].corr(integrated_data['total_sold']),
                        'Impressões vs Vendas': integrated_data['impressions'].corr(integrated_data['total_sold']),
                        'Clicks vs Vendas': integrated_data['clicks'].corr(integrated_data['total_sold']),
                        'Investimento vs Receita': integrated_data['spend'].corr(integrated_data['sales_to_date'])
                    }
                    
                    corr_df = pd.DataFrame(list(correlations.items()), columns=['Métrica', 'Correlação'])
                    corr_df['Correlação'] = corr_df['Correlação'].round(3)
                    corr_df['Força'] = corr_df['Correlação'].apply(
                        lambda x: 'Forte' if abs(x) >= 0.7 else 'Moderada' if abs(x) >= 0.4 else 'Fraca'
                    )
                    
                    st.dataframe(corr_df, use_container_width=True)

def main():
    """Função principal do aplicativo"""
    
    # Cabeçalho
    st.markdown('<h1 class="main-header">📊 Ads Analyzer v2.0</h1>', unsafe_allow_html=True)
    st.markdown("*Análise integrada de performance de anúncios e dados de vendas com mapeamento detalhado*")
    
    # Sidebar
    st.sidebar.header("⚙️ Configurações")
    
    # Inicializar conectores e processadores
    sheets_connector = PublicSheetsConnector()
    ads_processor = AdsDataProcessor()
    dashboard = IntegratedDashboard()
    
    # Seção 1: Dados de Vendas (Google Sheets Público)
    st.sidebar.subheader("📊 Dados de Vendas")
    
    load_sales = st.sidebar.button("🔄 Carregar Dados de Vendas", key="load_sales")
    
    if load_sales or 'sales_data' not in st.session_state:
        with st.spinner("Carregando dados de vendas..."):
            sales_data = sheets_connector.load_data()
            if sales_data is not None:
                st.session_state.sales_data = sales_data
                st.sidebar.success(f"✅ {len(sales_data)} shows carregados")
                
                # Mostrar resumo dos dados
                summary = sheets_connector.get_data_summary(sales_data)
                st.sidebar.info(f"📈 Ocupação média: {summary.get('avg_occupancy', 0):.1f}%")
                st.sidebar.info(f"🏙️ {summary.get('unique_cities', 0)} cidades")
            else:
                st.sidebar.error("❌ Erro ao carregar dados")
                st.session_state.sales_data = None
    
    # Seção 2: Dados de Anúncios
    st.sidebar.subheader("📈 Dados de Anúncios")
    
    ads_file = st.sidebar.file_uploader(
        "Upload dos dados de anúncios",
        type=['csv', 'xlsx', 'xls'],
        help="Arquivo com dados de performance de anúncios (Meta, Google, etc.)"
    )
    
    if ads_file:
        try:
            # Carregar arquivo
            if ads_file.name.endswith('.csv'):
                ads_data = pd.read_csv(ads_file)
            else:
                ads_data = pd.read_excel(ads_file)
            
            # Processar dados
            ads_data = ads_processor.detect_and_normalize_columns(ads_data)
            ads_data = ads_processor.calculate_missing_kpis(ads_data)
            
            st.session_state.ads_data = ads_data
            st.sidebar.success(f"✅ {len(ads_data)} registros de anúncios carregados")
            
        except Exception as e:
            st.sidebar.error(f"❌ Erro ao processar arquivo: {str(e)}")
            st.session_state.ads_data = None
    
    # Dashboard principal
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Vendas", "📈 Anúncios", "🔗 Integração", "🔍 Dados Brutos"])
    
    with tab1:
        if 'sales_data' in st.session_state and st.session_state.sales_data is not None:
            dashboard.create_sales_overview(st.session_state.sales_data)
            st.markdown("---")
            dashboard.create_sales_charts(st.session_state.sales_data)
        else:
            st.info("👆 Carregue os dados de vendas usando o botão na barra lateral")
    
    with tab2:
        if 'ads_data' in st.session_state and st.session_state.ads_data is not None:
            dashboard.create_ads_overview(st.session_state.ads_data)
            st.markdown("---")
            dashboard.create_ads_charts(st.session_state.ads_data)
        else:
            st.info("👆 Faça upload dos dados de anúncios usando a barra lateral")
    
    with tab3:
        sales_df = st.session_state.get('sales_data')
        ads_df = st.session_state.get('ads_data')
        
        if sales_df is not None and ads_df is not None:
            # Preparar dados para integração
            ads_df, sales_df, mapping = ads_processor.prepare_for_integration(ads_df, sales_df)
            dashboard.create_integration_analysis(sales_df, ads_df, mapping)
        else:
            st.info("👆 Carregue tanto os dados de vendas quanto os dados de anúncios para ver a análise integrada")
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎫 Dados de Vendas")
            if 'sales_data' in st.session_state and st.session_state.sales_data is not None:
                st.dataframe(st.session_state.sales_data, use_container_width=True)
                
                # Opção de download
                csv = st.session_state.sales_data.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV - Vendas",
                    csv,
                    "sales_data.csv",
                    "text/csv"
                )
            else:
                st.info("Nenhum dado de vendas carregado")
        
        with col2:
            st.subheader("📈 Dados de Anúncios")
            if 'ads_data' in st.session_state and st.session_state.ads_data is not None:
                st.dataframe(st.session_state.ads_data, use_container_width=True)
                
                # Opção de download
                csv = st.session_state.ads_data.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV - Anúncios",
                    csv,
                    "ads_data.csv",
                    "text/csv"
                )
            else:
                st.info("Nenhum dado de anúncios carregado")
    
    # Footer
    st.markdown("---")
    st.markdown("*Desenvolvido com ❤️ usando Streamlit | Ads Analyzer v2.0 - Integração Completa*")

if __name__ == "__main__":
    main()