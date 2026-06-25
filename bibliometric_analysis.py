#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from wordcloud import WordCloud
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BibliometricAnalyzer:
    def __init__(self, csv_path):
        
        self.df = pd.read_csv(csv_path)
        self.df_filtered = None
        self.brazil_df = None
        self.international_df = None
        
    def load_and_clean_data(self):
        
        print("Loading and cleaning data...")
        print(f"Original dataset shape: {self.df.shape}")
        
        # Filter years (exclude 2014, work with 2015-2025)
        self.df_filtered = self.df[self.df['PY'].between(2015, 2025)].copy()
        print(f"Filtered dataset shape (2015-2025): {self.df_filtered.shape}")
        
        # Clean and process key columns
        self.df_filtered['AU_clean'] = self.df_filtered['AU'].fillna('')
        self.df_filtered['TI_clean'] = self.df_filtered['TI'].fillna('')
        self.df_filtered['AB_clean'] = self.df_filtered['AB'].fillna('')
        self.df_filtered['KW_clean'] = self.df_filtered['KW_Merged'].fillna('')
        self.df_filtered['SO_clean'] = self.df_filtered['SO'].fillna('')
        self.df_filtered['C1_clean'] = self.df_filtered['C1'].fillna('')
        
        # Identify Brazilian studies: by author/institution affiliation (C1) only.
        # Previously also matched biome keywords (Amazon/Cerrado/etc.) in the
        # abstract/title/keywords, which wrongly flagged studies with zero
        # Brazilian affiliation (e.g. Peruvian/Colombian Amazon research) as
        # "Brazilian" just for discussing a transboundary biome.
        self.df_filtered['is_brazilian'] = self.df_filtered['C1_clean'].str.contains('BRAZIL', case=False, na=False)
        
        self.brazil_df = self.df_filtered[self.df_filtered['is_brazilian']].copy()
        self.international_df = self.df_filtered[~self.df_filtered['is_brazilian']].copy()
        
        print(f"Brazilian studies: {len(self.brazil_df)}")
        print(f"International studies: {len(self.international_df)}")
        
        return self.df_filtered
    
    def extract_authors(self, author_string):
        
        if pd.isna(author_string) or author_string == '':
            return []
        authors = [author.strip() for author in author_string.split(';')]
        return [author for author in authors if author]
    
    def extract_keywords(self, keyword_string):
        
        if pd.isna(keyword_string) or keyword_string == '':
            return []
        keywords = [kw.strip().upper() for kw in keyword_string.split(';')]
        return [kw for kw in keywords if kw and len(kw) > 2]
    
    def analyze_authors(self):
        
        print("Analyzing authors...")
        
        # Extract all authors
        all_authors = []
        for authors_str in self.df_filtered['AU_clean']:
            all_authors.extend(self.extract_authors(authors_str))
        
        author_counts = Counter(all_authors)
        top_authors = pd.DataFrame(author_counts.most_common(20), 
                                 columns=['Author', 'Publications'])
        
        # Brazilian authors
        brazil_authors = []
        for authors_str in self.brazil_df['AU_clean']:
            brazil_authors.extend(self.extract_authors(authors_str))
        
        brazil_author_counts = Counter(brazil_authors)
        top_brazil_authors = pd.DataFrame(brazil_author_counts.most_common(15), 
                                        columns=['Author', 'Publications'])
        
        return top_authors, top_brazil_authors
    
    def analyze_journals(self):
        
        print("Analyzing journals...")
        
        journal_counts = self.df_filtered['SO_clean'].value_counts().head(20)
        top_journals = pd.DataFrame({
            'Journal': journal_counts.index,
            'Publications': journal_counts.values
        })
        
        # Brazilian studies journals
        brazil_journal_counts = self.brazil_df['SO_clean'].value_counts().head(15)
        top_brazil_journals = pd.DataFrame({
            'Journal': brazil_journal_counts.index,
            'Publications': brazil_journal_counts.values
        })
        
        return top_journals, top_brazil_journals
    

    def analyze_keywords(self):
       
        print("Analyzing keywords...")
        
        all_keywords = self.df_filtered['DE'].dropna().str.split(';').explode().str.strip()
        top_keywords = all_keywords.value_counts().head(10).reset_index()
        top_keywords.columns = ['Keyword', 'Count']

        top_brazil_keywords = pd.DataFrame(columns=['Keyword', 'Count'])
        if hasattr(self, 'df_brazil') and not self.df_brazil.empty:
            brazil_keywords = self.df_brazil['DE'].dropna().str.split(';').explode().str.strip()
            if not brazil_keywords.empty:
                top_brazil_keywords = brazil_keywords.value_counts().head(10).reset_index()
                top_brazil_keywords.columns = ['Keyword', 'Count']

        print("Generating improved word cloud...")
        prepared_words = self._prepare_words_for_cloud(self.df_filtered['DE'])
        fig_wordcloud, top_words_for_cloud_df = self._plot_wordcloud_figure(prepared_words)
        
        # Retorna todos os resultados, incluindo a nova nuvem de palavras e a tabela de palavras
        return top_keywords, top_brazil_keywords, fig_wordcloud, top_words_for_cloud_df
    
    def analyze_temporal_evolution(self):
        """Analyze temporal evolution of publications"""
        print("Analyzing temporal evolution...")
        
        # Overall temporal evolution
        yearly_counts = self.df_filtered['PY'].value_counts().sort_index()
        
        # Brazilian vs International
        brazil_yearly = self.brazil_df['PY'].value_counts().sort_index()
        intl_yearly = self.international_df['PY'].value_counts().sort_index()
        
        # Combine into DataFrame
        years = range(2015, 2026)
        temporal_df = pd.DataFrame({
            'Year': years,
            'Total': [yearly_counts.get(year, 0) for year in years],
            'Brazilian': [brazil_yearly.get(year, 0) for year in years],
            'International': [intl_yearly.get(year, 0) for year in years]
        })
        
        return temporal_df
    
    def extract_countries_from_affiliations(self, affiliation_string):
        """Extract countries from affiliation string"""
        if pd.isna(affiliation_string) or affiliation_string == '':
            return []
        
        # Common country patterns in affiliations
        country_patterns = {
            'USA': ['USA', 'UNITED STATES', 'AMERICA'],
            'CHINA': ['CHINA', 'PEOPLES R CHINA'],
            'BRAZIL': ['BRAZIL', 'BRASIL'],
            'GERMANY': ['GERMANY', 'DEUTSCHLAND'],
            'FRANCE': ['FRANCE'],
            'ITALY': ['ITALY', 'ITALIA'],
            'SPAIN': ['SPAIN', 'ESPANA'],
            'CANADA': ['CANADA'],
            'AUSTRALIA': ['AUSTRALIA'],
            'UK': ['ENGLAND', 'UNITED KINGDOM', 'UK', 'SCOTLAND', 'WALES'],
            'INDIA': ['INDIA'],
            'JAPAN': ['JAPAN'],
            'SOUTH KOREA': ['SOUTH KOREA', 'KOREA'],
            'NETHERLANDS': ['NETHERLANDS', 'HOLLAND'],
            'SWEDEN': ['SWEDEN'],
            'NORWAY': ['NORWAY'],
            'FINLAND': ['FINLAND'],
            'DENMARK': ['DENMARK'],
            'SWITZERLAND': ['SWITZERLAND'],
            'AUSTRIA': ['AUSTRIA'],
            'BELGIUM': ['BELGIUM'],
            'PORTUGAL': ['PORTUGAL'],
            'MEXICO': ['MEXICO'],
            'ARGENTINA': ['ARGENTINA'],
            'CHILE': ['CHILE'],
            'COLOMBIA': ['COLOMBIA'],
            'PERU': ['PERU'],
            'ECUADOR': ['ECUADOR'],
            'VENEZUELA': ['VENEZUELA']
        }
        
        countries = []
        affiliation_upper = affiliation_string.upper()

        # Word-boundary match: plain substring search wrongly matched e.g.
        # "UK" inside "DUKE UNIVERSITY" or "GERMANY" patterns inside longer
        # institution names with no relation to the country.
        for country, patterns in country_patterns.items():
            for pattern in patterns:
                if re.search(r'\b' + re.escape(pattern) + r'\b', affiliation_upper):
                    countries.append(country)
                    break

        return list(set(countries))  # Remove duplicates
    
    def analyze_geographic_distribution(self):
        """Analyze geographic distribution of publications"""
        print("Analyzing geographic distribution...")
        
        # Extract countries from affiliations
        all_countries = []
        for affiliation in self.df_filtered['C1_clean']:
            all_countries.extend(self.extract_countries_from_affiliations(affiliation))
        
        country_counts = Counter(all_countries)
        geographic_df = pd.DataFrame(country_counts.most_common(20), 
                                   columns=['Country', 'Publications'])
        
        return geographic_df
    
    def identify_data_types(self, text):
        """Identify data types mentioned in text"""
        if pd.isna(text):
            return []
        
        text_upper = text.upper()
        data_types = []
        
        data_patterns = {
            'SATELLITE IMAGERY': ['SATELLITE', 'LANDSAT', 'SENTINEL', 'MODIS', 'SPOT'],
            'LIDAR': ['LIDAR', 'LiDAR', 'LASER SCANNING', 'ALS'],
            'DRONE/UAV': ['DRONE', 'UAV', 'UAS', 'UNMANNED AERIAL', 'RPAS'],
            'HYPERSPECTRAL': ['HYPERSPECTRAL', 'HYPERSPEC'],
            'MULTISPECTRAL': ['MULTISPECTRAL', 'MULTISPEC'],
            'RADAR': ['RADAR', 'SAR', 'SYNTHETIC APERTURE'],
            'OPTICAL': ['OPTICAL', 'VISIBLE', 'NIR', 'SWIR'],
            'THERMAL': ['THERMAL', 'TIR', 'TEMPERATURE'],
            'FIELD MEASUREMENTS': ['FIELD MEASUREMENT', 'GROUND TRUTH', 'IN-SITU', 'FIELD DATA'],
            'SPECTRORADIOMETER': ['SPECTRORADIOMETER', 'SPECTROMETER'],
            'PHOTOGRAMMETRY': ['PHOTOGRAMMETRY', 'SfM', 'STRUCTURE FROM MOTION'],
            'METEOROLOGICAL': ['METEOROLOGICAL', 'WEATHER', 'CLIMATE DATA'],
            'TOPOGRAPHIC': ['TOPOGRAPHIC', 'DEM', 'DIGITAL ELEVATION', 'DTM']
        }
        
        # Word-boundary match: bare substrings like "ALS" or "TIR" matched
        # inside common words ("materials", "entire"), wildly inflating
        # LiDAR/Thermal counts.
        for data_type, patterns in data_patterns.items():
            for pattern in patterns:
                if re.search(r'\b' + re.escape(pattern) + r'\b', text_upper):
                    data_types.append(data_type)
                    break

        return list(set(data_types))

    def analyze_ai_techniques(self):
        """
        Analyzes AI techniques for overall frequency and trends over time.
        """
        print("Analyzing AI techniques with improved dictionary...")

        ai_terms = {
            'Random Forest': ['random forest', 'randomforest'],
            'Support Vector Machine': ['support vector machine', 'svm', 'support vector regression', 'svr'],
            'Neural Networks': ['neural network', 'ann', 'artificial neural network'],
            'Deep Learning': ['deep learning', 'aprendizado profundo'],
            'Convolutional NN': ['convolutional neural network', 'cnn'],
            'Recurrent NN': ['recurrent neural network', 'rnn'],
            'LSTM': ['long short-term memory', 'lstm'],
            'GRU': ['gated recurrent unit', 'gru'],
            'Transformer': ['transformer', 'transformadores'],
            'U-Net': ['u-net', 'unet'],
            'GAN': ['generative adversarial network', 'gan'],
            'Autoencoder': ['autoencoder', 'auto-encoder'],
            'Gradient Boosting': ['gradient boosting', 'gbm', 'xgboost', 'lightgbm', 'catboost'],
            'k-Nearest Neighbors': ['k-nearest neighbors', 'knn'],
            'Regression Models': [
                'linear regression', 'multiple regression', 'multivariate regression',
                'logistic regression', 'stepwise regression', 'partial least squares regression',
                'pls regression', 'pls-r', 'regression tree', 'polynomial regression',
                'nonlinear regression', 'non-linear regression', 'geographically weighted regression',
                'regression kriging', 'ridge regression', 'lasso regression', 'regression model',
                'regression analysis', 'regression algorithm', 'regression equation', 'regressão',
            ],
            'Cubist': ['cubist'],
            'Attention Mechanism': ['attention mechanism'],
        }

        # Word-boundary match: bare substrings like "ann" or "gan" matched
        # inside "channel"/"annual"/"organic", inflating Neural Networks/GAN.
        def find_terms(text, terms_dict):
            found_terms = set()
            if pd.isna(text): return list(found_terms)
            text_lower = text.lower()
            for term, patterns in terms_dict.items():
                for pattern in patterns:
                    if re.search(r'\b' + re.escape(pattern) + r'\b', text_lower):
                        found_terms.add(term)
                        break
            return list(found_terms)

        text_columns = ['TI', 'AB', 'DE']
        for col in text_columns:
            if col in self.df_filtered.columns:
                self.df_filtered[col] = self.df_filtered[col].fillna('')
        self.df_filtered['combined_text'] = self.df_filtered[text_columns].agg(' '.join, axis=1)
        self.df_filtered['AI_Techniques'] = self.df_filtered['combined_text'].apply(lambda text: find_terms(text, ai_terms))

        df_exploded_all = self.df_filtered.explode('AI_Techniques').dropna(subset=['AI_Techniques'])
        ai_counts = df_exploded_all['AI_Techniques'].value_counts()
        ai_techniques_df = ai_counts.head(15).reset_index()
        ai_techniques_df.columns = ['Technique', 'Count']

        print("Analyzing AI technique trends over time...")
        df_exploded_trends = self.df_filtered.explode('AI_Techniques').dropna(subset=['AI_Techniques', 'PY'])
        top_7_techniques = ai_techniques_df['Technique'].head(7).tolist()
        ai_trends_grouped = df_exploded_trends[df_exploded_trends['AI_Techniques'].isin(top_7_techniques)]
        ai_trends_df = ai_trends_grouped.groupby(['PY', 'AI_Techniques']).size().reset_index(name='Count')

        return ai_techniques_df, ai_trends_df

    def analyze_data_types(self):
        """Analyze data types usage"""
        print("Analyzing data types...")
        
        # Extract data types
        all_data_types = []
        data_type_by_year = {}
        
        for idx, row in self.df_filtered.iterrows():
            data_types = self.identify_data_types(row['combined_text'])
            all_data_types.extend(data_types)
            
            year = row['PY']
            if year not in data_type_by_year:
                data_type_by_year[year] = []
            data_type_by_year[year].extend(data_types)
        
        # Overall data type frequency
        data_type_counts = Counter(all_data_types)
        data_types_df = pd.DataFrame(data_type_counts.most_common(15), 
                                   columns=['Data_Type', 'Frequency'])
        
        # Data type trends over time
        data_type_trends = {}
        for year in range(2015, 2026):
            year_data_types = Counter(data_type_by_year.get(year, []))
            for data_type in data_type_counts.keys():
                if data_type not in data_type_trends:
                    data_type_trends[data_type] = []
                data_type_trends[data_type].append(year_data_types.get(data_type, 0))
        
        trends_df = pd.DataFrame(data_type_trends)
        trends_df['Year'] = range(2015, 2026)
        
        return data_types_df, trends_df
    
    def analyze_drone_usage(self):
        """Analyze drone/UAV usage"""
        print("Analyzing drone/UAV usage...")
        
        # Identify studies using drones
        drone_keywords = ['DRONE', 'UAV', 'UAS', 'UNMANNED AERIAL', 'RPAS']
        
        self.df_filtered['uses_drone'] = self.df_filtered['combined_text'].str.contains(
            '|'.join(drone_keywords), case=False, na=False
        )
        
        drone_studies = self.df_filtered[self.df_filtered['uses_drone']].copy()
        
        # Drone usage by year
        drone_by_year = drone_studies['PY'].value_counts().sort_index()
        
        # Most promising data for drone use. Excludes "DRONE/UAV" itself: this
        # subset is already filtered to drone studies, so that category would
        # trivially be #1 every time and add no information.
        drone_data_types = []
        for text in drone_studies['combined_text']:
            drone_data_types.extend(self.identify_data_types(text))
        drone_data_types = [d for d in drone_data_types if d != 'DRONE/UAV']

        drone_data_counts = Counter(drone_data_types)
        promising_drone_data = pd.DataFrame(drone_data_counts.most_common(10),
                                          columns=['Data_Type', 'Frequency'])
        
        return len(drone_studies), drone_by_year, promising_drone_data

    @staticmethod
    def _top_cited(df, n=20):
        def truncate(title, max_len=70):
            if not isinstance(title, str) or not title:
                return 'Untitled'
            return title if len(title) <= max_len else title[:max_len].rstrip() + '…'

        df = df.copy()
        df['TC'] = pd.to_numeric(df['TC'], errors='coerce').fillna(0)
        top = df.sort_values('TC', ascending=False).head(n)[['TI', 'TC', 'PY']].copy()
        top['TI_short'] = top['TI'].apply(truncate)
        return top.reset_index(drop=True)

    def analyze_most_cited(self):
        """Most cited documents in the current filtered selection (TC field)."""
        print("Analyzing most cited documents...")
        return self._top_cited(self.df_filtered)

    def analyze_most_cited_brazil(self):
        """Most cited documents among the Brazilian-affiliated subset."""
        print("Analyzing most cited Brazilian documents...")
        return self._top_cited(self.brazil_df)


    def run_analysis(self, df_to_analyze):
        """
        Runs all bibliometric analyses on a given DataFrame.
        """
        print("Running analysis on the filtered dataframe...")

        self.df_filtered = df_to_analyze
        self.brazil_df = self.df_filtered[self.df_filtered['is_brazilian']]
        self.international_df = self.df_filtered[~self.df_filtered['is_brazilian']]

        top_authors, top_brazil_authors = self.analyze_authors()
        top_journals, top_brazil_journals = self.analyze_journals()
        top_keywords, top_brazil_keywords, fig_wordcloud, top_words_df = self.analyze_keywords()
        temporal_df = self.analyze_temporal_evolution()
        geographic_df = self.analyze_geographic_distribution()
        ai_techniques_df, ai_trends_df = self.analyze_ai_techniques()
        data_types_df, data_trends_df = self.analyze_data_types()
        drone_count, drone_by_year, promising_drone_data = self.analyze_drone_usage()
        drone_areas_evolution, drone_areas_recent = self.analyze_drone_study_areas()
        cooc_ia_area, cooc_data_area = self.analyze_cooc_matrices()
        most_cited_df = self.analyze_most_cited()
        most_cited_brazil_df = self.analyze_most_cited_brazil()

        # Armazena todos os resultados no dicionário
        results = {
            'most_cited_df': most_cited_df,
            'most_cited_brazil_df': most_cited_brazil_df,
            'top_authors': top_authors,
            'top_brazil_authors': top_brazil_authors,
            'top_journals': top_journals,
            'top_brazil_journals': top_brazil_journals,
            'top_keywords': top_keywords,
            'top_brazil_keywords': top_brazil_keywords,
            'temporal_df': temporal_df,
            'geographic_df': geographic_df,
            'ai_techniques_df': ai_techniques_df,
            'ai_trends_df': ai_trends_df,
            'data_types_df': data_types_df,
            'data_trends_df': data_trends_df,
            'drone_count': drone_count,
            'drone_by_year': drone_by_year,
            'promising_drone_data': promising_drone_data,
            'wordcloud_fig': fig_wordcloud,
            'top_words_df': top_words_df,
            'drone_areas_evolution': drone_areas_evolution,
            'drone_areas_recent': drone_areas_recent,
            'cooc_ia_area': cooc_ia_area,
            'cooc_data_area': cooc_data_area
        }
        
        return results

    def _prepare_words_for_cloud(self, series):
        """
        Processa o texto para gerar uma lista de palavras limpas para a nuvem.
        """
        # Dicionário para unificar termos. Fique à vontade para customizar!
        keyword_map = {
            'remote sensing technology': 'remote sensing',
            'satellite imagery': 'remote sensing',
            'satellite remote sensing': 'remote sensing',
            'uav': 'drone',
            'unmanned aerial vehicle': 'drone',
            'unmanned aerial vehicles': 'drone',
            'carbon stocks': 'carbon stock',
            'models': 'model',
            'modeling': 'model',
            'neural network': 'neural networks',
            'deep learning model': 'deep learning',
            'machine learning model': 'machine learning'
        }
        # Palavras a serem ignoradas
        stopwords = set(['article', 'study', 'studies', 'method', 'methods', 'approach', 'based', 'using', 'application'])

        all_words = []
        # Usar .copy() para evitar o SettingWithCopyWarning
        series_copy = series.copy()
        series_copy.dropna(inplace=True)

        for index, value in series_copy.items():
            words = [kw.strip().lower() for kw in str(value).split(';')]
            cleaned_words = []
            for word in words:
                word = keyword_map.get(word, word)
                if word and word not in stopwords and len(word) > 2:
                    cleaned_words.append(word)
            all_words.extend(cleaned_words)
        return all_words

    def _plot_wordcloud_figure(self, processed_words):
        """
        Gera a figura da nuvem de palavras e um DataFrame com os termos mais frequentes.
        """
        if not processed_words:
            return None, None
            
        word_counts = Counter(processed_words)
        top_words_df = pd.DataFrame(word_counts.most_common(20), columns=['Term', 'Frequency'])

        wordcloud = WordCloud(
            width=1200, height=600, background_color='white', colormap='viridis',
            max_words=100, contour_width=3, contour_color='steelblue',
            collocations=False, random_state=42
        ).generate_from_frequencies(word_counts)

        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        return fig, top_words_df    

    def analyze_drone_study_areas(self):
        """
        Analyzes the application of drones across different study areas over time.
        """
        print("Analyzing drone applications in study areas...")

        # Definição das palavras-chave para a análise
        drone_keywords = ["drone", "uav", "unmanned aerial vehicle"]
        areas_keywords = {
            "Environment": r'\benvironment(al)?\b',
            "Restoration": r'\brestoration\b',
            "Conservation": r'\bconservation\b',
            "Degradation": r'\bdegradation\b',
            "Agriculture": r'\bagriculture\b',
            "Aquatic": r'\baquatic\b',
            "Forestry": r'\bforestry\b',
            "Urban": r'\burban\b',
            "Wetland": r'\bwetland\b',
            "Grassland": r'\bgrassland\b',
            "Biodiversity": r'\bbiodiversity\b'
        }

        # Prepara a coluna de texto combinado
        text_columns = ['TI', 'AB', 'DE']
        for col in text_columns:
            if col in self.df_filtered.columns:
                self.df_filtered[col] = self.df_filtered[col].fillna('')
        self.df_filtered['combined_text'] = self.df_filtered[text_columns].agg(' '.join, axis=1).str.lower()
        
        # Filtra o DataFrame para artigos que mencionam drones
        df_drones = self.df_filtered[self.df_filtered["combined_text"].str.contains("|".join(drone_keywords), na=False)].copy()

        if df_drones.empty:
            return pd.DataFrame(), pd.DataFrame()

        # --- Análise 1: Evolução ao Longo do Tempo ---
        # Conta as menções de cada área por ano
        for area, pattern in areas_keywords.items():
            df_drones[area] = df_drones["combined_text"].str.contains(pattern, na=False, flags=re.IGNORECASE)
        
        evolution_df = df_drones.groupby("PY")[list(areas_keywords.keys())].sum()
        evolution_df = evolution_df.reset_index()
        # Transforma o DataFrame para o formato 'long' para ser compatível com o Plotly
        evolution_df_long = evolution_df.melt(id_vars='PY', var_name='Study Area', value_name='Publications')


        # --- Análise 2: Destaque nos Últimos 5 Anos ---
        current_year = pd.Timestamp.now().year
        recent_years_drones = df_drones[df_drones['PY'] >= current_year - 5]
        
        recent_highlights = recent_years_drones[list(areas_keywords.keys())].sum().sort_values(ascending=False).reset_index()
        recent_highlights.columns = ['Study Area', 'Publications']

        return evolution_df_long, recent_highlights
    


    def analyze_cooc_matrices(self):
        """
        Analyzes the co-occurrence of AI techniques and data types with study areas.
        """
        print("Analyzing co-occurrence matrices...")

        # Definição das palavras-chave
        study_areas = {
            "Environment": r'\benvironment(al)?\b', "Restoration": r'\brestoration\b',
            "Conservation": r'\bconservation\b', "Degradation": r'\bdegradation\b',
            "Agriculture": r'\bagriculture\b', "Aquatic": r'\baquatic\b',
            "Forestry": r'\bforestry\b', "Urban": r'\burban\b',
            "Wetland": r'\bwetland\b', "Grassland": r'\bgrassland\b',
            "Biodiversity": r'\bbiodiversity\b'
        }
        
        ai_techniques = {
            'Random Forest': ['random forest', 'randomforest'],
            'Support Vector Machine': ['support vector machine', 'svm', 'support vector regression', 'svr'],
            'Neural Networks': ['neural network', 'ann', 'artificial neural network'],
            'Deep Learning': ['deep learning', 'aprendizado profundo'],
            'Convolutional NN': ['convolutional neural network', 'cnn'],
            'Recurrent NN': ['recurrent neural network', 'rnn'],
            'LSTM': ['long short-term memory', 'lstm'],
            'GRU': ['gated recurrent unit', 'gru'],
            'Transformer': ['transformer', 'transformadores'],
            'U-Net': ['u-net', 'unet'],
            'GAN': ['generative adversarial network', 'gan'],
            'Autoencoder': ['autoencoder', 'auto-encoder'],
            'Gradient Boosting': ['gradient boosting', 'gbm', 'xgboost', 'lightgbm', 'catboost'],
            'k-Nearest Neighbors': ['k-nearest neighbors', 'knn'],
            'Regression Models': [
                'linear regression', 'multiple regression', 'multivariate regression',
                'logistic regression', 'stepwise regression', 'partial least squares regression',
                'pls regression', 'pls-r', 'regression tree', 'polynomial regression',
                'nonlinear regression', 'non-linear regression', 'geographically weighted regression',
                'regression kriging', 'ridge regression', 'lasso regression', 'regression model',
                'regression analysis', 'regression algorithm', 'regression equation', 'regressão',
            ],
            'Cubist': ['cubist'],
            'Attention Mechanism': ['attention mechanism'],
        }
        
        data_types = {
            "UAV/Drone": r'\bdrone|uav|unmanned aerial vehicle\b',
            "LiDAR": r'\blidar|light detection and ranging\b',
            "Satellite Imagery": r'\bsatellite|landsat|sentinel|modis|remote sensing\b',
            "Hyperspectral": r'\bhyperspectral\b',
            "Multispectral": r'\bmultispectral\b',
            "SAR": r'\bsar|synthetic aperture radar\b',
            "Field Data": r'\bfield data|ground truth|in-situ\b'
        }

        if 'combined_text' not in self.df_filtered.columns or self.df_filtered['combined_text'].isnull().all():
            text_columns = ['TI', 'AB', 'DE']
            for col in text_columns:
                if col in self.df_filtered.columns:
                    self.df_filtered[col] = self.df_filtered[col].fillna('')
            self.df_filtered['combined_text'] = self.df_filtered[text_columns].agg(' '.join, axis=1).str.lower()

        cooc_ia_area_list = []
        for area_name, area_pattern in study_areas.items():
            df_area = self.df_filtered[self.df_filtered["combined_text"].str.contains(area_pattern, na=False, flags=re.IGNORECASE)]
            if not df_area.empty:
                for tech_name, tech_patterns in ai_techniques.items():
                    pattern_str = r'\b(' + '|'.join(map(re.escape, tech_patterns)) + r')\b'
                    count = df_area["combined_text"].str.contains(pattern_str, na=False, flags=re.IGNORECASE).sum()
                    if count > 0:
                        cooc_ia_area_list.append({"Study Area": area_name, "AI Technique": tech_name, "Count": count})
        
        pivot_ia_areas = pd.DataFrame()
        if cooc_ia_area_list:
            df_ia_area = pd.DataFrame(cooc_ia_area_list)
            pivot_ia_areas = df_ia_area.pivot_table(index="AI Technique", columns="Study Area", values="Count").fillna(0).astype(int)

        cooc_data_area_list = []
        for area_name, area_pattern in study_areas.items():
            df_area = self.df_filtered[self.df_filtered["combined_text"].str.contains(area_pattern, na=False, flags=re.IGNORECASE)]
            if not df_area.empty:
                for data_name, data_pattern in data_types.items():
                    count = df_area["combined_text"].str.contains(data_pattern, na=False, flags=re.IGNORECASE).sum()
                    if count > 0:
                        cooc_data_area_list.append({"Study Area": area_name, "Data Type": data_name, "Count": count})

        pivot_dados_areas = pd.DataFrame()
        if cooc_data_area_list:
            df_data_area = pd.DataFrame(cooc_data_area_list)
            pivot_dados_areas = df_data_area.pivot_table(index="Data Type", columns="Study Area", values="Count").fillna(0).astype(int)

        return pivot_ia_areas, pivot_dados_areas
    


if __name__ == "__main__":
    analyzer = BibliometricAnalyzer('DF_COMBINADO_LIMPO.csv')
    base_df = analyzer.load_and_clean_data()
    results = analyzer.run_analysis(base_df)
    
    print("\n" + "="*50)
    print("BIBLIOMETRIC ANALYSIS RESULTS")
    print("="*50)
    
    print(f"\nDataset Overview:")
    print(f"- Total studies (2015-2025): {len(analyzer.df_filtered)}")
    print(f"- Brazilian studies: {len(analyzer.brazil_df)}")
    print(f"- International studies: {len(analyzer.international_df)}")
    print(f"- Studies using drones/UAV: {results['drone_count']}")
    
    print(f"\nTop 10 Authors:")
    print(results['top_authors'].head(10).to_string(index=False))
    
    print(f"\nTop 10 Journals:")
    print(results['top_journals'].head(10).to_string(index=False))
    
    print(f"\nTop 15 Keywords:")
    print(results['top_keywords'].head(15).to_string(index=False))
    
    print(f"\nTop AI Techniques:")
    print(results['ai_techniques_df'].to_string(index=False))
    
    print(f"\nTop Data Types:")
    print(results['data_types_df'].to_string(index=False))
    
    print(f"\nMost Promising Drone Data:")
    print(results['promising_drone_data'].to_string(index=False))
    
    print(f"\nGeographic Distribution (Top 10):")
    print(results['geographic_df'].head(10).to_string(index=False))
