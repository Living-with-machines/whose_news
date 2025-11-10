import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import jensenshannon
import seaborn as sns
from multiprocessing import Pool
import re

def count_vector(counts):
    sums = counts.sum(axis=0) 
    return np.array(sums).flatten()

def prob_vector(counts):
    sums = counts.sum(axis=0)
    total = sums.sum()
    return np.array(sums / total).flatten()

def to_distribution(categories,index):
    counter = Counter(categories)
    counts = pd.Series(counter,index=index)
    prob = counts / np.sum(counts)
    return prob

def to_penalized_distribution(categories,index, penalty=100):
  
    counter = Counter(categories)
    counts = pd.Series(counter,index=index)
    
    props_p = counts / (counts.sum() + (counts*penalty))
    prob_pen_norm = props_p / props_p.sum()
    return prob_pen_norm

def to_uniform_distribution(categories,index):
    counter = {c:1/len(index) for c in categories}
    counts = pd.Series(counter,index=index)
    prob = counts / np.sum(counts)
    return prob


def to_softmax_distribution(categories,index, temperature=.5):
    counter = Counter(categories)
    counts = pd.Series(counter,index=index)
    nom = np.exp(counts/temperature)
    denom = nom.sum()
    return nom / denom

class FightingWordsMini(object):
    def __init__(self,df,p1='liberal',p2='conservative', content='S-TITLE', column='Leaning'):
        self.df = df
        self._p1 = p1
        self._p2 = p2
        self.content = content
        self.column = column
        
    def get_counts(self):
        self.vectorizer = CountVectorizer(ngram_range=(1,3), min_df=5, stop_words='english' )
        self.counts = self.vectorizer.fit_transform(self.df[self.content ].astype(str))
        self.vocab_array = self.vectorizer.get_feature_names_out()
        
    def get_indices(self):
        self.idxdict = {self._p1: list(self.df[self.df[self.column]==self._p1].idx.values),
                        self._p2: list(self.df[self.df[self.column]==self._p2].idx.values)
                       }
        
    
    def count_matrix(self,p):
        return np.matrix(count_vector(self.counts[self.idxdict[p]]))

    def create_count_matrices(self):
        self.p1 = self.count_matrix(self._p1)
        self.p2 = self.count_matrix(self._p2)

    def compute_totals(self):
        self.n_p1 = self.p1.sum(axis=1)
        self.n_p2 = self.p2.sum(axis=1)

    def run(self, a0=1e-5):
        self.get_counts()
        self.get_indices()
        
        self.create_count_matrices()
        self.compute_totals()

        a0_sum = a0*self.p1.shape[1]
   
        u_p1 = (self.p1+a0) / (self.n_p1+a0_sum)
        u_p2 = (self.p2+a0) / (self.n_p2+a0_sum)

        log_odds_matrix = np.log( u_p1 / (1 - u_p1)) - np.log( u_p2 / (1 - u_p2))
        var_log_odds_matrix = (1 / (self.p1  + a0)) + (1 / (self.p2  + a0))
        std_log_odds = log_odds_matrix / np.sqrt(var_log_odds_matrix)
        self.fw = np.array(std_log_odds).flatten()
        
    def get_fw_score_for_word(self,word):
        pass
        
    def most_biased_by_pattern(self,regex,top_n=25, return_results=False):

        
        
        sorted_vocab_array = self.vocab_array[np.argsort(self.fw)]
        sorted_scores_array = self.fw[np.argsort(self.fw)]
        filtered_vocab_array = np.array([f'{w}, {round(sorted_scores_array[i],2)}' for i,w in enumerate(sorted_vocab_array) if re.match(regex,w)])
        
        if return_results:
            return {self._p1: filtered_vocab_array[-top_n:][::-1], self._p2:filtered_vocab_array[:top_n]}

        print(f'Most {self._p2}\n','; '.join(filtered_vocab_array[:top_n]))
        print('\n')
        print(f'Most {self._p1}\n','; '.join(filtered_vocab_array[-top_n:][::-1]))

class FightingWords(object):
    def __init__(self, corpus,
                       p1: str = 'liberal',  # positive end of contrast dimension 
                       p2: str = 'conservative',  # negative end of contrast dimension
                       facet : str = 'Leaning',  # dimension to contrast
                       ts: str = 'year',  # dimension for timeline
                       add_levels = [], # dimension to stratify
                       start: int = 1830,  # start of time dimension 
                       end: int = 1920, #  end of time dimension
                       sample_size : int = 2, # sample size for comparison
                       runs : int = 100 # repeat the computation n times
                       ):
        self.corpus = corpus
        self._p1 = p1
        self._p2 = p2 
        self.facet = facet 
        self.ts = ts 
        self.add_levels = add_levels 
        self.start = start 
        self.end = end 
        self.time_filter = (corpus.metadata.year.between(self.start,self.end))
        self.sample_size = sample_size 
        self.runs = runs
    
    def set_grouped_indices(self):
        #mask = self.corpus.metadata[self.facet].isin([self._p1,self._p2])
        mask = self.corpus.metadata[self.facet].isin([self._p1,self._p2]) & (self.corpus.metadata.year.between(self.start,self.end))
        #print(sum(mask) / self.corpus.metadata.shape[0])
        levels = [self.facet,self.ts] + self.add_levels
        self.grouped_indices = self.corpus.metadata[mask].groupby(levels)['idx'].apply(lambda x: list(x))
        # drop rows for which we can not make a comparison across the facet dimension
        #print(collected_ids)
        # print(f'Using {round(collected_ids / self.corpus.metadata.shape[0]*100,2)} per cent of observations before dropping nan rows.')

        self.grouped_indices = self.grouped_indices.unstack(self.facet).dropna(axis=0).reset_index()

        collected_ids = self.grouped_indices.apply(
            lambda x : len(x[self._p1])+ len(x[self._p2]), axis=1
                ).sum()#.plot(kind='density')
        #print(collected_ids)
        print(f'Using {round(collected_ids / self.corpus.metadata.shape[0]*100,2)} per cent of observations after filtering.')


    def sample_indices(self):
        self.sampled_indices = pd.concat([
                self.grouped_indices.apply(
                    #lambda x, p: random.sample(x[p],min(self.sample_size,len(x[p])) ), 
                    lambda x, p: random.sample(x[p], max(1, int(len(x[p])*self.sample_size))) , 
                                        p=p, axis=1) # ADD MIN HERE!!! If p > number of observations
                                            for p in [self._p1,self._p2]], axis=1)


        self.sampled_indices.columns =  [self._p1,self._p2]
        #print(self.sampled_indices)
        self.sampled_indices = pd.concat([self.grouped_indices[[self.ts] + self.add_levels] , self.sampled_indices], axis=1)
        
    def simplify_political_labels(self, classification_csv, target_col, new_col='Leaning'):
        pol_labels = pd.read_csv(classification_csv, index_col=0)
        self.convert_dict = pol_labels[['label_orig',target_col]].set_index('label_orig',drop=True).to_dict()[target_col]
        self.corpus.metadata[new_col] = self.corpus.metadata['S-POL'].replace(self.convert_dict).astype(str)


    def count_matrix(self,p):
        return np.matrix([count_vector(self.corpus.counts[self.sampled_indices.loc[i,p]])
                          for i in range(self.sampled_indices.shape[0])])

    def create_count_matrices(self):
        self.p1 = self.count_matrix(self._p1)
        self.p2 = self.count_matrix(self._p2)

    def compute_totals(self):
        self.n_p1 = self.p1.sum(axis=1)
        self.n_p2 = self.p2.sum(axis=1)

    def run(self, a0=1e-5):
        self.sample_indices()
        self.create_count_matrices()
        self.compute_totals()

        a0_sum = a0*self.p1.shape[1]
   
        u_p1 = (self.p1+a0) / (self.n_p1+a0_sum)
        u_p2 = (self.p2+a0) / (self.n_p2+a0_sum)

        log_odds_matrix = np.log( u_p1 / (1 - u_p1)) - np.log( u_p2 / (1 - u_p2))
        var_log_odds_matrix = (1 / (self.p1  + a0)) + (1 / (self.p2  + a0))
        std_log_odds = log_odds_matrix / np.sqrt(var_log_odds_matrix)
        self.fw = std_log_odds

    def compute_fighting_words(self):
        
        self.set_grouped_indices()
        self.sampling_results = []
        
        for _ in tqdm(range(self.runs)):
            self.run()
            self.sampling_results.append(self.fw)
        
        self.matrix_concatenated = np.concatenate(self.sampling_results,axis=0)

    def compute_lexical_bias(self, divide_by_std=False):
        nominator = np.array(self.matrix_concatenated.mean(axis=0)).flatten()
        
        if divide_by_std:
            denominator =  np.array(self.matrix_concatenated.std(axis=0)).flatten()
            self.lexical_bias = nominator / (1+denominator)
        
        else:
            self.lexical_bias = nominator #/ denominator
       # self.stable_bias[np.where((nominator < .05) & (nominator > -.05) )] = 0

    def most_biased_by_pattern(self,regex,top_n=25, return_results=False,save_to=False):

        if not hasattr(self,'lexical_bias'):
            self.compute_lexical_bias()

        vocab_array = np.array(list(self.corpus.vocab.keys()))
        sorted_vocab_array = vocab_array[np.argsort(self.lexical_bias)]
        sorted_scores_array = self.lexical_bias[np.argsort(self.lexical_bias)]
        filtered_vocab_array = np.array([f'{w}, {round(sorted_scores_array[i],2)}' for i,w in enumerate(sorted_vocab_array) if re.match(regex,w)])
        
        if save_to:
            path = Path(save_to)
            path.mkdir(exist_ok=True)
            scores_df = pd.DataFrame(np.array([(w, float(round(sorted_scores_array[i],3))) for i,w in enumerate(sorted_vocab_array) if re.match(regex,w)]),
                                        columns=['words','score']
                                            )
            scores_df.score = scores_df.score.astype(float) 
            scores_df.sort_values('score',inplace = True, ascending=False)
            scores_df.reset_index(drop=True, inplace = True)
            scores_df_top_n = scores_df.iloc[list(range(top_n)) + list(range(scores_df.shape[0]-top_n,scores_df.shape[0]))]
            
            scores_df_top_n.reset_index(drop=True, inplace=True)
            scores_df_top_n['code'] = 0
            scores_df_top_n.to_csv(path / f'fw_scores_{self._p1}_{self._p2}_{self.start}_{self.end}_{bool(self.facet)}.csv')
            
            return scores_df_top_n

        if return_results:
            return {self._p1: filtered_vocab_array[-top_n:][::-1], self._p2:filtered_vocab_array[:top_n]}

        print(f'Most {self._p2}\n','; '.join(filtered_vocab_array[:top_n]))
        print('\n')
        print(f'Most {self._p1}\n','; '.join(filtered_vocab_array[-top_n:][::-1]))

    def plot_query(self,query,invert=False,average=False):
        scores = []
        for r in self.sampling_results:
            if average:
                scores.append(np.array(r[:,[self.corpus.vocab[q] for q in query]].mean(axis=1)).flatten())
            else:
                scores.append(np.array(r[:,[self.corpus.vocab[q] for q in query]].sum(axis=1)).flatten())
            
        df = pd.DataFrame(np.matrix(scores).T,
                #index=[y for y in range(self.start,self.end) if y in self.grouped_indices.index])
                index=[y for y in self.grouped_indices[self.ts]])
        df_long = df.stack().reset_index()
        df_long.columns = ['year','sample','fw_score']
        #return df_long
        if invert:
            df_long.fw_score = df_long.fw_score * -1.0
        sns.set(rc={'figure.figsize':(12.7,5.27)})
        g =  sns.lineplot(x='year',y='fw_score',data= df_long)
        g.set_xticks(range(self.start,self.end,5)) # <--- set the ticks first
        g.set_xticklabels(range(self.start,self.end,5))

    def save_sampling_results(self, save_to):
        path = Path(save_to)
        path.mkdir(exist_ok=True)
        print(path / f'fw_scores_{self._p1}_{self._p2}.npy')
        with open(path / f'fw_scores_{self._p1}_{self._p2}.npy', 'wb') as f:
            np.save(f,self.sampling_results)

    def load_sampling_results(path):
        with open(f'{path}/fw_scores_{self._p1}_{self._p2}.npy', 'rb') as f:
            self.sampling_results = np.load(f)
        self.matrix_concatenated = np.concatenate(self.sampling_results,axis=0)



class DataBlender(object):
    """Main class used for the representativeness analysis.
    Relies on several inputs, therefore called the DataBlender
    """

    def __init__(self, directories: pd.DataFrame = None, 
                        digital_papers: pd.DataFrame= None, 
                        annotations: pd.DataFrame = None):
        """
        Arguments:
            directories (pd.DataFrame)
            digitial_papers
            annotations
        """
        self.directories = directories
        self.digital_papers = digital_papers
        self.annotations = annotations

        nlp_years = self.digital_papers.groupby('NLP')['year'].apply(lambda x: set().union(*x)).reset_index()
        self.digital_papers_by_year = nlp_years.explode('year').reset_index()

    @classmethod
    def load_data(cls, path_dir: str, path_paper: str, path_anno: str):
        """Load all dataframes and pass them to the object
        """
        directories = pd.read_csv(path_dir, index_col=0)
        
        overview = pd.read_csv(path_paper)
        overview = overview[~overview.AcquiredYears.isnull()]
        overview['year'] = overview['AcquiredYears'].apply(lambda x: [int(i) for i in x.strip().split()] if isinstance(x,str) else [])

        annotations = pd.read_csv(path_anno, index_col=0)
        annotations = annotations[annotations.label=='yes']
        
        return cls(directories, overview, annotations)

    

    def recode_political_labels(self, classification_csv, target_col):
        """Given a mapping, simplify/recode a the political labels
        """
        pol_labels = pd.read_csv(classification_csv, index_col=0)
        self.pol_convert_dict = pol_labels[['label_orig',target_col]].set_index('label_orig',drop=True).to_dict()[target_col]
        self.directories['S-POL'] = self.directories['S-POL'].apply(lambda x: str(x).strip())
        
        self.directories['Leaning'] = self.directories['S-POL'].replace(self.pol_convert_dict)
    
    def complete_records_in_directories(self, target_col):
        """This function replaces nan values based on the temporarily
        closest record of the same newspapers (i.e. newspapers with
        the same chain_id)
        """

        print('Before: ',sum(~self.directories[target_col].isnull()))
        self.directories = self.complete_records(self.directories,target_col)
        print('After: ',sum(~self.directories[target_col].isnull()))

    def complete_records(self, df: pd.DataFrame, target_col:str) -> pd.DataFrame:
        """function to add information to the press directories detaframe. we use the 
        chain_id column to add missing information, for example if, for one year, a 
        newspaper doesn't have a S-POL we insert one that is closest in time from a row with
        the same chain_id"""

        df = df.copy()
        df[f'value_{target_col}_source_idx'] = None
        df_chain = df[~df.chain_id.isnull()]
        chains = set(df_chain.chain_id.unique())
    
        for chain_id in chains:
            # get elements for a specific chain_id
            chain_df = df[df.chain_id==chain_id]
            # find those the have NaA values
            no_attr = list(np.where(chain_df[target_col].isnull())[0])

            # if there are empty but not all values are empty
            if no_attr and not (len(no_attr) == chain_df.shape[0]): 
                # look which cells have a value for this columns       
                has_attr = np.where(~chain_df[target_col].isnull())[0]
                # find indices for cells (that have content) closest to an empty cell
                replace_with = chain_df.iloc[[has_attr[np.argmin(abs(has_attr - i))] for i in no_attr]][target_col]
                zipped = list(zip(chain_df.iloc[no_attr].index,replace_with.index,replace_with.values))
                for target_idx, source_idx, cat in zipped:
                    df.loc[target_idx,target_col] = cat
                    df.loc[target_idx,f'value_{target_col}_source_idx'] = source_idx

        return df


    def plot_corpus_timeline(self, with_links=False, with_directories=False, start=1780, end=1920, s_x=10, s_y = 5):
        """Function to plot the size of the different datasets over time.
        """

        ax =  self.digital_papers_by_year[ 
                    self.digital_papers_by_year.year.isin(range(start,end))
                            ].groupby('year')['NLP'
                                ].count().plot(figsize=(s_x,s_y), style=['-'])
        
        if with_links:
            self.digital_papers_by_year['has_npd_link'] = False
           
            found = set(self.annotations.NLP) 
            for i, row in self.digital_papers.iterrows():
                if row.NLP in found:
                     self.digital_papers_by_year.loc[ 
                                self.digital_papers_by_year.NLP==row.NLP,'has_npd_link'
                                                        ] = True

            #digital_papers[digital_papers.year.isin(range(1780,1920))].groupby('year')['NLP'].count().plot(figsize=(10,5))
            ax =  self.digital_papers_by_year[ 
                    self.digital_papers_by_year.year.isin(range(start,end)) & (self.digital_papers_by_year.has_npd_link == True)
                                                ].groupby('year')['NLP'].count().plot(style=['--']) # figsize=(7,6) # color=['grey','orange'],
        ax.axvline(1846, color='k', linestyle='--')
        if with_directories:
            self.directories.groupby('year')['id'].count().plot(style=['-.'])
        return ax
        


    def add_directories_to_newspapers(self):
        """Combine Mitchell with the newspaper overview spreadsheet
        We add an npd_id to each newspaper year (i.e. digital_newspaper_by_year)
        """
        def obs_by_ids(ids):
            """
            if a newspaper could not be linked to a chain, we linked
            it to MPD_ ie. unique identifier in the directories
            there we get all rows that either match the chain id 
            or unique id.
            """
            cid = {c for c in ids if c.startswith('CID_')}
            nid = {c for c in ids if c.startswith('MPD_')}
    
            sub_df = self.directories[self.directories.chain_id.isin(cid) | self.directories.id.isin(nid) ]
            return sub_df.reset_index()

        self.digital_papers_by_year['npd_id'] = None
        found = set(self.annotations.NLP) 

        missing_ids = set()
        for i, row in self.digital_papers_by_year.iterrows():
            if row.NLP in found:
              
                ids = set(self.annotations[self.annotations.NLP==row.NLP].chain_id)
                
                sub_df = obs_by_ids(ids)
                if sub_df.shape[0] > 0:
                    closest_row_id = np.argmin(np.abs(row.year - sub_df.year))
                    
                    closest_obs = sub_df.iloc[closest_row_id]
                    self.digital_papers_by_year.loc[i,'npd_id'] = closest_obs.id
                else:
                    missing_ids.update(ids)
        print(missing_ids)
               
                

    def merge_directories_and_newspapers(self):
        """First add the npd_id to each row in digital_papers_by_year
        Then merge the press directories with and newspaper overview, i.e.
        for each year we have a digitised papers, at the most relevant entry
        from the press directories
        """
        self.add_directories_to_newspapers()
        self.data = self.digital_papers_by_year.merge(self.directories,left_on='npd_id',right_on='id', how='left',suffixes=['','_npd'])

    def distributions_by_year(self, column, include_values, start=1846, end=1920):
        """get the yearly distributions of a set of values
        recode all others as Other
        """
        def _label_distribution(target='newspapers'):
            if target == 'newspapers':
                data = self.data
            elif target == 'directories':
                data  = self.directories
            else:
                return 'Error'
            if column not in data.columns:
                return 'Error: column not in dataframe'
            
            
            data_sel = data[data.year.between(start,end)].copy()
            data_sel.loc[~data_sel[column].isin(include_values),column] = 'other'
            print(data_sel[column].value_counts(),data_sel[column].unique())
            return data_sel.groupby(['year',column])['id'].count() / data_sel.groupby(['year'])['id'].count()
           
        distribution_newspapers = _label_distribution('newspapers').unstack(column)
        distribution_directories = _label_distribution('directories').unstack(column)
        
        return distribution_newspapers,distribution_directories
        
    def compare_proportions_for_labels(self, column, target_labels, colors, include_values, **kwargs):
        """function mainly used for plotting and comparing the distribution of 
        labels over time
        """
        distribution_newspapers,distribution_directories = self.distributions_by_year(column, include_values)
        
        fig, ax = plt.subplots(figsize=(6,5))

        for col, tl in zip(colors,target_labels):
            ax.plot(distribution_directories.index,
                    distribution_directories[tl],
                    linestyle='dotted',
                    marker = '^',
                    linewidth=2,
                    color=col
                    
                    )
            
            ax.plot(distribution_newspapers.index,
                    distribution_newspapers[tl],
                    linestyle='dashed',
                    #marker = 'o',
                    color=col,
                    alpha=.6
                    )

        return ax


    def add_chrono_nlp(self):
        """rank NLPs in ascending order starting from zero
        """
        self.data['NLP_chron'] = self.data.NLP.apply(lambda x: x if x < 5000 else -1*x)
        nlp_to_sort_id = {nlp: i for i, nlp in enumerate(sorted(self.data['NLP_chron'].unique()))}
        self.data.replace({'NLP_chron': nlp_to_sort_id}, inplace=True)

    def compare_target_distributions(self,field, selected_labels):
        """compute the "ideal" distribution for different definitions of
        representativeness and return them as a dataframe. this is used to 
        visually inspect what these may look like.
        """
        directories_red = self.directories.copy()
        directories_red.loc[~directories_red[field].isin(selected_labels), field] = 'other'
        #data_red = self.data[self.data[field].isin(selected_labels)]
        index = sorted(directories_red[field].unique())
        
        proportional = to_distribution(directories_red[field],index)
        equal = to_uniform_distribution(directories_red[field],index)
        penalized = to_penalized_distribution(directories_red[field],index)
        return pd.DataFrame({'proportional':proportional, 'equal': equal, 'weighted': penalized})
        
    def bias_over_time(self, distribution_sample,distribution_population,method=None, p=100, col='Leaning'):
        """compute the yearly bias for each edition where we have press directories
        """
        timeline = sorted(self.directories.year.unique())

        if method == 'equal':
            n_col = distribution_population.shape[1]
            distribution_population = pd.DataFrame({year : [1 / n_col]*n_col for year in timeline}).T
            distribution_population.columns = list(distribution_sample.columns)
        if method == 'penalized':
            counts = self.directories[self.directories[col].isin(distribution_population.columns)
                    ].groupby(['year',col])['id'].count()
            totals = counts.groupby(level=0).transform('sum')
            props_p = counts / (totals + counts*p)
            distribution_population = (props_p / props_p.groupby(level=0).transform('sum')).unstack()
            
        distribution_population.fillna(.0, inplace=True)
        distribution_sample.fillna(.0, inplace=True)
       
        bias_dict = {
                    year:
                        jensenshannon(distribution_sample.loc[year],
                                    distribution_population.loc[year]
                                    )
                            for year in timeline
                        }

       
        bias_contribution_dict = defaultdict(dict) 
        for label in distribution_population.columns:
            for year in timeline:
                p = distribution_sample.loc[year,label]
                q =  distribution_population.loc[year,label]
                bias_contribution_dict[label][year] = p*np.log(2*p / (p+q))
            



        #return pd.Series(bias_dict,index=timeline)#.plot()
        return bias_dict,bias_contribution_dict,timeline,distribution_population
        
    def bias_digitization(self, field, selected_labels, method, step_size = 10):
        """compute bias as a function of digitization over time. we first sort
        the nlp identifiers in chronological order and then compute bias from 
        0 till moment n with n increases determined by step size.
        """
        if method == 'proportional':
            method = to_distribution
        elif method == 'equal':
            method = to_uniform_distribution
        elif method == 'reweighted':
            method = to_penalized_distribution

        results = {}
        #directories_red = self.directories[self.directories[field].isin(selected_labels)].copy
        #data_red = self.data[self.data[field].isin(selected_labels)]
        directories_red = self.directories.copy()
        directories_red.loc[~directories_red[field].isin(selected_labels),field] = 'other'
        data_red = self.data.copy()
        data_red.loc[~data_red[field].isin(selected_labels),field] = 'other'
        #print(directories_red[field].value_counts())
        #print(data_red[field].value_counts())
        index = sorted(directories_red[field].unique())

        pop_distribution = method(directories_red[field],index)

        for i in range(0,data_red.NLP_chron.max(),step_size):
            sample = data_red[data_red.NLP_chron.between(0,i+step_size)]
            sample_distribution = to_distribution(sample[field],index)
    
            results[i] = jensenshannon(sample_distribution,pop_distribution)
        return pd.Series(results)

    def bias_digitization_by_value(self, field, target, 
                                    target_dist='proportional', 
                                    method='diff', 
                                    selected= None, 
                                    batch_level = False,
                                    window = 200,
                                    step_size = 50):
        """plot which groups are driving the digitization bias 
        """
        results = {}
    
        if not selected:
            selected = list(self.directories[field].unique())
        directories_red = self.directories[self.directories[field].isin(selected)]
        dpm_red = self.data[self.data[field].isin(selected)]
    
        index = sorted(directories_red[field].unique())

        if target_dist == 'proportional':
            pop_distribution = to_distribution(self.directories[field], index)
            target_pop = pop_distribution[target]
        elif target_dist == 'equal':
            target_pop = 1 / len(selected)

        for i in range(0,dpm_red.NLP_chron.max(),step_size):
            
            if batch_level:
                start_at_idx = i
                
            else:
                start_at_idx = 0
            sample = dpm_red[dpm_red.NLP_chron.between(start_at_idx,i+window)]
            sample_distribution = to_distribution(sample[field],index)
            target_sample = sample_distribution[target]
            if method == 'diff':
                results[i] = target_pop - target_sample
            elif method == 'kl':
                results[i] = target_sample * np.log(2* target_sample / (target_pop + target_sample))
        
        return pd.Series(results)

    @property
    def directories_by_title(self):
        """prepare data for regression analysis
        """
        annotations_unique = self.annotations.drop_duplicates(subset=['NLP','chain_id'])

        most_frequent = lambda x, column: sorted(Counter(x[column]).items(),key=lambda y: y[1],reverse=True)[0][0]
        pol = self.directories.groupby('chain_id').apply(most_frequent, column='Leaning')
        county = self.directories.groupby('chain_id').apply(most_frequent, column='COUNTY')
        #num_obs = lambda x, column: len(x[column])
        run_length = self.directories.groupby('chain_id').apply(len)
        directories_by_title =  pd.concat([pol, county,run_length], axis=1).reset_index()
        directories_by_title['digital'] = 0
        directories_by_title['digital'] = directories_by_title.chain_id.apply(
                                                lambda x: 1 if x in self.annotations.chain_id.values else 0
                                                )
        directories_by_title.columns = ['chain_id','pol','county','run_length','digital']

        directories_by_title = directories_by_title.merge(self.annotations[['chain_id','NLP']],right_on='chain_id',left_on='chain_id', suffixes=['','_anno'], how='left')
        directories_by_title['NLP_chron'] = directories_by_title.NLP.apply(lambda x: x if x < 5000 else -1*x)
        nlp_to_sort_id = {nlp: i for i, nlp in enumerate(sorted(directories_by_title['NLP_chron'].unique()))}
        directories_by_title.replace({'NLP_chron': nlp_to_sort_id}, inplace=True)

        return directories_by_title



# ------------ OLD CODE FOR COMPUTING THE FIGHTING WORDS ------------
# ------- functionalities replaced by FightingWords class -----------


class FightingWordsTimeline(object):
    def __init__(self, corpus,
                       p1: str = 'liberal', 
                       p2: str = 'conservative', 
                       facet : str = 'Leaning',
                       ts: str = 'year',
                       start: int = 1830, 
                       end: int = 1920,
                       
                       ):
        self.corpus = corpus
        self._p1 = p1
        self._p2 = p2
        self.facet = facet
        self.ts = ts
        #self.select_facets = [self._p1,self._p2]
        self.start = start
        self.end = end
        self.time_filter = (corpus.metadata.year.between(self.start,self.end))
        
        #self.set_grouped_indices()

    def set_grouped_indices(self,add_level=False):
        mask = self.corpus.metadata[self.facet].isin([self._p1,self._p2]) & (self.corpus.metadata.year.between(self.start,self.end))
        print(sum(mask))
        if add_level:
            levels = [self.facet,add_level,self.ts]
        else:
            levels = [self.facet,self.ts]
        self.grouped_indices = self.corpus.metadata[mask].groupby(levels)['idx'].apply(lambda x: list(x))
        #return grouped_indices

    def simplify_political_labels(self, classification_csv, target_col, new_col='Leaning'):
        pol_labels = pd.read_csv(classification_csv, index_col=0)
        self.convert_dict = pol_labels[['label_orig',target_col]].set_index('label_orig',drop=True).to_dict()[target_col]
        self.corpus.metadata['S-POL'] = self.corpus.metadata['S-POL'].apply(lambda x: str(x).strip())
        self.corpus.metadata[new_col] = self.corpus.metadata['S-POL'].replace(self.convert_dict).astype(str)

    def count_matrix(self,p):
        return np.matrix([count_vector(self.corpus.counts[self.grouped_indices[p, year]])
                          for year in tqdm(range(self.start,self.end))])

    def create_count_matrices(self):
        self.p1 = self.count_matrix(self._p1)
        self.p2 = self.count_matrix(self._p2)

    def compute_totals(self):
        self.n_p1 = self.p1.sum(axis=1)
        self.n_p2 = self.p2.sum(axis=1)


    def compute_fighting_words(self, a0=1e-5):
        self.set_grouped_indices()
        self.create_count_matrices()
        self.compute_totals()

        a0_sum = a0*self.p1.shape[1]
   
        u_p1 = (self.p1+a0) / (self.n_p1+a0_sum)
        u_p2 = (self.p2+a0) / (self.n_p2+a0_sum)

        log_odds_matrix = np.log( u_p1 / (1 - u_p1)) - np.log( u_p2 / (1 - u_p2))
        var_log_odds_matrix = (1 / (self.p1  + a0)) + (1 / (self.p2  + a0))
        std_log_odds = log_odds_matrix / np.sqrt(var_log_odds_matrix)
        self.fw = std_log_odds

    def timeline(self,query_term):
        s1 = pd.Series(np.array((self.p1 / self.n_p1)[:,self.corpus.vocab[query_term]]).flatten())
        s2 = pd.Series(np.array((self.p2 / self.n_p2)[:,self.corpus.vocab[query_term]]).flatten())
        query_df = pd.concat([s1,s2], axis=1)
        query_df.columns = [self._p1, self._p2]
        return query_df.plot()

  
    def bias_timeline(self, query_term):
        return pd.Series(np.array(self.fw[:,self.corpus.vocab[query_term]]).flatten(), 
          index=range(self.start,self.end)
            ).plot(title=query_term.upper())

    def stable_bias(self, top_n=50):
        scores = np.array(self.fw.mean(axis=0)).flatten() / np.array(self.fw.std(axis=0)).flatten()
        vocab_array = np.array(list(self.corpus.vocab))
        sorted_vocab_array = vocab_array[np.argsort(scores)]
        print(f'Most {self._p1}',sorted_vocab_array[-top_n:])
        print(f'Most {self._p2}',sorted_vocab_array[:top_n])

    def most_biased_by_year(self, year,top_n=50):
        std_log_odds_flat = np.array(self.fw[year - self.start,:]).flatten()
        vocab_array = np.array(list(self.corpus.vocab))
        sorted_vocab_array = vocab_array[np.argsort(std_log_odds_flat)]
        print(f'Most {self._p1}',sorted_vocab_array[-top_n:])
        print(f'Most {self._p2}',sorted_vocab_array[:top_n])

class FightingWordsSampled(FightingWordsTimeline):

    def __init__(self, corpus, sample_size, **kwargs):
        self.sample_size = sample_size
        FightingWordsTimeline.__init__(self,corpus, **kwargs)

    def set_grouped_indices(self, add_level=False):
        import random
        random.seed(27101984)
        super().set_grouped_indices(add_level)
        if add_level:
            #levels = ['year',add_level,'Leaning']
            df = self.grouped_indices.unstack('Leaning').dropna(axis=0)
        else:
            levels = ['year','Leaning']
            df = self.grouped_indices.groupby(['year','Leaning']).sum().unstack('Leaning').dropna(axis=0)
        
        #return df
        #df.dropna(axis=0,inplace=True)
        self.grouped_indices = pd.concat([df.apply(lambda x, p: random.sample(x[p],min(self.sample_size,len(x[p])) ),p=p,axis=1) # ADD MIN HERE!!! If p > number of observations
                                                     for p in [self._p1,self._p2]], axis=1)

        self.grouped_indices.columns =  [self._p1,self._p2]
    
    def count_matrix(self,p):
        
        return np.matrix([count_vector(self.corpus.counts[self.grouped_indices.loc[y,p]])
                          #for y in )
                          for y in range(self.start,self.end) if y in self.grouped_indices.index])

    
    def plot_query(self,query,invert=False):
        scores = []
        for r in self.sampling_results:
            scores.append(np.array(r[:,[self.corpus.vocab[q] for q in query]].sum(axis=1)).flatten())
            
        df = pd.DataFrame(np.matrix(scores).T,
                index=[y for y in range(self.start,self.end) if y in self.grouped_indices.index])
        df_long = df.stack().reset_index()
        df_long.columns = ['year','sample','fw_score']
        if invert:
            df_long.fw_score = df_long.fw_score * -1.0
        sns.set(rc={'figure.figsize':(16.7,5.27)})
        return sns.lineplot(x='year',y='fw_score',data= df_long)

    def repeated_sampling(self,runs=100):
        # parallellize later
        self.sampling_results = []
        for _ in tqdm(range(runs)):
            self.compute_fighting_words()
            self.sampling_results.append(self.fw)
        
        self.matrix_concatenated = np.concatenate(self.sampling_results,axis=0)
        

    def compute_stable_bias(self):
        nominator = np.array(self.matrix_concatenated.mean(axis=0)).flatten()
        denominator =  np.array(self.matrix_concatenated.std(axis=0)).flatten()
        
        self.stable_bias = nominator #/ denominator
       # self.stable_bias[np.where((nominator < .05) & (nominator > -.05) )] = 0

    def manipulate_bias_scores(self, min_cutoff, max_cutoff, min_fw):
        self.stable_bias[np.where((self.stable_bias < min_fw) & (self.stable_bias > min_fw) )] = 0
        
        p1_totals = np.array(self.p1.sum(axis=0)).flatten()
        max_cutoff = sorted(np.array(self.p1.sum(axis=0)).flatten())[max_cutoff]
        
        self.stable_bias[np.where((p1_totals < min_cutoff) & (p1_totals > max_cutoff))[0]] = 0

    def most_biased_terms(self, top_n=25):
        vocab_array = np.array(list(self.corpus.vocab.keys()))
        print(f'Most {self._p1}',vocab_array[self.stable_bias.argsort()[-top_n:]][::-1])
        print(f'Most {self._p2}',vocab_array[self.stable_bias.argsort()[:top_n]])

    def most_biased_by_pattern(self,regex,top_n=25, return_results=False):
        vocab_array = np.array(list(self.corpus.vocab.keys()))
        sorted_vocab_array = vocab_array[np.argsort(self.stable_bias)]
        filtered_vocab_array = np.array([w for w in sorted_vocab_array if re.match(regex,w)])
        
        if return_results:
            return {self._p1: filtered_vocab_array[-top_n:][::-1], self._p2:filtered_vocab_array[:top_n]}

        print(f'Most {self._p2}',filtered_vocab_array[:top_n])
        print('\n')
        print(f'Most {self._p1}',filtered_vocab_array[-top_n:][::-1])

class FightingWordsStratified(FightingWordsTimeline):
    def __init__(self, corpus, level, **kwargs):
        self.level = level
        FightingWordsTimeline.__init__(self,corpus, **kwargs)

    
    def select_observations(self):
        observation_frequency = self.corpus.metadata[self.corpus.metadata[self.facet].isin([self._p1,self._p2])
                                 ].groupby([self.level,self.ts]).apply(lambda x: len(set(x[self.facet])))

        self.select_rows = observation_frequency[observation_frequency > 1].index

    def set_grouped_indices(self):
        self.select_observations()
        self.grouped_indices = self.corpus.metadata[self.corpus.metadata[self.facet].isin([self._p1,self._p2])
                                 ].groupby([self.level,self.ts,self.facet])['idx'].apply(lambda x: list(x))
        
        self.grouped_indices = self.grouped_indices.reset_index().set_index([self.level,self.ts]).loc[ self.select_rows].reset_index()
        self.grouped_indices = self.grouped_indices.pivot(index=[self.level,self.ts],columns=self.facet,values='idx').reset_index()
        self.grouped_indices.sort_values([self.level,self.ts], inplace=True)
        self.grouped_indices.reset_index(inplace=True)

    def count_matrix(self,p):
        return np.matrix([count_vector(self.corpus.counts[self.grouped_indices.loc[i,p]])
                          for i in tqdm(range(self.grouped_indices.shape[0]))])

    #def compute_totals(self):
    #    self.n_p1 = self.p1.sum(axis=1)[...,None]
    #    self.n_p2 = self.p2.sum(axis=1)[...,None]

    def backup(self,path):
        path = Path(path)
        path.mkdir(exist_ok=True)
        self.grouped_indices.to_csv(path / f'fw_{self._p1}_{self._p2}_stratified_metadata.csv')

        with open(path / f'{self._p1}_{self._p2}_count_matrix.npy', 'wb') as f:
            np.save(f, [self.p1,self.p2])

    def start_from_backup(self,path):
        path = Path(path)
        with open(path / f'{self._p1}_{self._p2}_count_matrix.npy', 'rb') as f:
            p1_count_matrix,p2_count_matrix = np.load(f)

        self.grouped_indices = pd.read_csv(path / f'fw_{self._p1}_{self._p2}_stratified_metadata.csv', index_col=0)
        
    def group_indices_by_timestep(self):
        self.indices_by_timestep = self.grouped_indices[self.grouped_indices.year.between(self.start,self.end)].groupby(self.ts)['index'].apply(list)

    def fighting_word_by_timestep(self):
        self.group_indices_by_timestep()
        self.fw_mean =  np.matrix([np.array(self.fw[idx,:].mean(axis=0)).flatten()
                            for ts, idx in self.indices_by_timestep.iteritems()])

    def bias_timeline(self, query_term):
        return pd.Series(np.array(self.fw_mean[:,self.corpus.vocab[query_term]]).flatten(), 
          index=sorted(set(self.grouped_indices.year))#range(self.start,self.end) # why adding one?
            ).plot(title=query_term.upper())

    def compute_stable_bias(self):
        """prefer words with high average fw scores but low std deviation over time
        """
        log_std = self.fw_mean.std(axis=0)
        log_sum = self.fw_mean.sum(axis=0)
        self.stable_bias = np.array(log_sum/log_std).flatten() # 

    def most_biased_terms(self,top_n=25):
        self.compute_stable_bias()
        
        vocab_array = np.array(list(self.corpus.vocab))
        sorted_vocab_array = vocab_array[np.argsort(self.stable_bias)]
        print(f'Most {self._p2}',sorted_vocab_array[:top_n])
        print('\n')
        print(f'Most {self._p1}',sorted_vocab_array[-top_n:][::-1])


    def most_biased_by_pattern(self,regex,top_n=25):
        self.compute_stable_bias()

        vocab_array = np.array(list(self.corpus.vocab))
        sorted_vocab_array = vocab_array[np.argsort(self.stable_bias)]
        filtered_vocab_array = [w for w in sorted_vocab_array if re.match(regex,w)]
        print(f'Most {self._p2}',filtered_vocab_array[:top_n])
        print('\n')
        print(f'Most {self._p1}',filtered_vocab_array[-top_n:][::-1])