import spacy
import scispacy
import pandas as pd
import numpy as np
import itertools
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

class TransformForExplorer:
    def __init__(self,abstractData_df,nlp):
        self.df = abstractData_df
        self.transformed_df = None
        self.nlp = nlp
        #self.entities_model = spacy.load("en_ner_bionlp13cg_md")
        self._transform_data()

    def _transform_data(self):
        self.transformed_df = self.df.copy()
        self._create_nlp_docs()
        self._vectors_from_docs()
        
        keep = ['title', 'abstract', 'journal',
       'publication_date', 'authors', 'docs','vectors'] 
        self.transformed_df = self.transformed_df.filter(keep,axis=1).dropna(
            subset=['docs','vectors'])

        self._entities_from_docs()
        return self.transformed_df

    def _create_nlp_docs(self):
        self.transformed_df['docs'] = self.transformed_df['abstract'].apply(
                                   self._safe_docs,args=((self.nlp,))) 

    def _vectors_from_docs(self):
        self.transformed_df['vectors'] = self.transformed_df['docs'].apply(
                                   self._safe_vectors,args=((self.nlp,))) 

    def _entities_from_docs(self):
        self.transformed_df['entities'] = self.transformed_df['abstract'].apply(
                            self.safe_entities,args=((self.nlp,))) 

        
    def safe_entities(self,x,entity_model):
        res_clean = []
        try:
            res = [i for i in itertools.chain(*entity_model(x).ents)]
            for i in res:
                if i.text.lower() not in res_clean:
                    res_clean.append(i.text.lower())
            return res_clean
        except Exception:
            return ["Failed"]

    def _safe_docs(self,x,nlp):
        try:
            return nlp(x)
        except Exception:
            return None

    def _safe_vectors(self,x,nlp):
        try:
            return x.vector
        except Exception:
            return None

class TransformForKG:
    def __init__(self,abstractData, nlp):
        # Only testing with 1 abstract
        #self.df = abstractData
        self.data = abstractData
        self.transformed_df = None
        self.nlp = nlp
        self.fig = None
        self.ax = None
        self.pairs = None
        self.kg_strings = None
        self._transform_data()
        

    def _transform_data(self):        
        #self.transformed_df = self.df.copy()
        self.transformed_data = self.data
        keep_columns = ['title', 'abstract']        
        #self.transformed_df = self.transformed_df.filter(keep_columns,axis=1).dropna(subset=keep_columns)
        self.transformed_data = self.data[keep_columns]
        pairs = self._create_entity_pairs(self.transformed_data['abstract'])
        #fig, ax = self._draw_KG(pairs)  
        #return self.transformed_df
        #self.fig = fig 
        #self.ax = ax

        self._manual_draw_KG(pairs)

    def _create_entity_pairs(self, text):
        #total_rows = self.transformed_df['abstract'].count()
        entity_pairs = []
        #for index in range(0, total_rows):
        rtn_pairs = self._get_entity_pairs(text)  #df['abstract'][index] if calling all abstracts in df
        for i in range(0, len(rtn_pairs)):
            entity_pairs.append([rtn_pairs['subject'][i], rtn_pairs['relation'][i], rtn_pairs['object'][i], 
                rtn_pairs['subject_type'][i], rtn_pairs['object_type'][i]])
        #print("Number of entity pairs ", len(rtn_pairs))
        entp_df = pd.DataFrame(entity_pairs, columns=['subject', 'relation', 'object',
                                             'subject_type', 'object_type'])
        return entp_df

    def _refine_ent(self, ent, sent):
        unwanted_tokens = (
            'PRON',  # pronouns
            'PART',  # particle
            'DET',  # determiner
            'SCONJ',  # subordinating conjunction
            'PUNCT',  # punctuation
            'SYM',  # symbol
            'X',  # other
            )
        ent_type = ent.ent_type_  # get entity type
        if ent_type == '':
            ent_type = 'NOUN_CHUNK'
            ent = ' '.join(str(t.text) for t in
                           self.nlp(str(ent)) if t.pos_
                           not in unwanted_tokens and t.is_stop == False)
        elif ent_type in ('NOMINAL', 'CARDINAL', 'ORDINAL') and str(ent).find(' ') == -1:
            refined = ''
            for i in range(len(sent) - ent.i):
                if ent.nbor(i).pos_ not in ('VERB', 'PUNCT'):
                    refined += ' ' + str(ent.nbor(i))
                else:
                    ent = refined.strip()
                    break

        return ent, ent_type
    
    def _get_entity_pairs(self, text):
        #sentences = [sent.string.strip() for sent in text.sents]  # split text into sentences
        sentences = [str(i) for i in self.nlp(text).sents]
        ent_pairs = []
        for sent in sentences:
            sent = self.nlp(sent)
            spans = list(sent.ents) + list(sent.noun_chunks)  # collect nodes
            spans = spacy.util.filter_spans(spans)
            with sent.retokenize() as retokenizer:
                [retokenizer.merge(span, attrs={'tag': span.root.tag,
                                            'dep': span.root.dep}) for span in spans]
            deps = [token.dep_ for token in sent]

            # limit our example to simple sentences with one subject and object
            if (deps.count('obj') + deps.count('dobj')) != 1\
                    or (deps.count('subj') + deps.count('nsubj')) != 1:
                continue

            for token in sent:
                if token.dep_ not in ('obj', 'dobj'):  # identify object nodes
                    continue
                subject = [w for w in token.head.lefts if w.dep_
                       in ('subj', 'nsubj')]  # identify subject nodes
                if subject:
                    subject = subject[0]
                    # identify relationship by root dependency
                    relation = [w for w in token.ancestors if w.dep_ == 'ROOT']
                    if relation:
                        relation = relation[0]
                        # add adposition or particle to relationship
                        if relation.nbor(1).pos_ in ('ADP', 'PART'):
                            relation = ' '.join((str(relation), str(relation.nbor(1))))
                    else:
                        relation = 'unknown'

                    subject, subject_type = self._refine_ent(subject, sent)
                    token, object_type = self._refine_ent(token, sent)

                    ent_pairs.append([str(subject), str(relation), str(token),
                                  str(subject_type), str(object_type)])

        ent_pairs = [sublist for sublist in ent_pairs
                          if not any(str(ent) == '' for ent in sublist)]
        pairs = pd.DataFrame(ent_pairs, columns=['subject', 'relation', 'object',
                                             'subject_type', 'object_type'])
        #print('Entity pairs extracted:', str(len(ent_pairs)))
        self.pairs = pairs
        #print(pairs)
        return pairs

    def _manual_draw_KG(self,pairs):
        if len(pairs) < 1: 
            return None
        res_list = []
        for i in range(pairs.shape[0]):
            res = f"{pairs.iloc[i].subject} {pairs.iloc[i].relation} {pairs.iloc[i].object}"
            res_list.append(res)
        self.kg_strings = res_list

    # def _draw_KG(self, pairs):
    #     if len(pairs) < 1: 
    #         return None,None
    #     k_graph = nx.from_pandas_edgelist(pairs, 'subject', 'object',
    #         create_using=nx.MultiDiGraph())
    #     node_deg = nx.degree(k_graph)
    #     layout = nx.spring_layout(k_graph, k=0.15, iterations=20)
    #     fig, ax = plt.subplots(figsize=(10, 10))
    #     # = plt.figure(num=None, figsize=(10, 10), dpi=80)
    #     nx.draw_networkx(
    #         k_graph,
    #         node_size=[int(deg[1]) * 500 for deg in node_deg],
    #         arrowsize=20,
    #         linewidths=1.5,
    #         pos=layout,
    #         edge_color='red',
    #         edgecolors='black',
    #         node_color='white',
    #     )
    #     labels = dict(zip(list(zip(pairs.subject, pairs.object)),
    #               pairs['relation'].tolist()))
    #     nx.draw_networkx_edge_labels(k_graph, pos=layout, edge_labels=labels,
    #                              font_color='red')
    #     ax.axis('off')
    #     return fig , ax