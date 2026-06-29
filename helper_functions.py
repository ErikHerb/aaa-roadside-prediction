from pyspark.sql.window import Window
import pandas as pd
import numpy as np
from pyspark.sql.functions import col, rank, when
from pyspark.sql.types import StringType, IntegerType, DoubleType, ArrayType, MapType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder

import scipy
scipy.interp=np.interp

import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def classification_performance(predDF):
  
  pd_prediction=predDF.select('label', 'prediction').toPandas()
  label=pd_prediction['label']
  pred=pd_prediction['prediction']
 
  confusion=confusion_matrix(label, pred)

  print('Confusion Matrix\n', confusion)

  print('\nClassification Report\n')

  print(classification_report(label, pred))
  
def prediction_performance(df):
    # Calculate overall accuracy
    total_count = df.count()
    accuracy = round(df.where("prediction == label").count() / total_count, 2)
    
    # Calculate true positives, false positives, true negatives, and false negatives
    tp = df.where("label = 1 and prediction = 1").count()
    fp = df.where("label = 0 and prediction = 1").count()
    tn = df.where("label = 0 and prediction = 0").count()
    fn = df.where("label = 1 and prediction = 0").count()
    
    # Calculate precision, recall, and specificity
    precision = round(tp / (tp + fp), 2) if tp + fp > 0 else None
    recall = round(tp / (tp + fn), 2) if tp + fn > 0 else None
    specificity = round(tn / (tn + fp), 2) if tn + fp > 0 else None
    
    # Calculate F1 score, considering the case where precision + recall is 0
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = round(2 * precision * recall / (precision + recall), 2)
    else:
        f1 = None
    
    # Return results as a dictionary
    return {
        'accuracy': accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        'f1_score': f1
    }

  # define a function to return feature names for logisitcs regression
def feature_names(df, features):
    feature_attrs = df.schema[features].metadata["ml_attr"]["attrs"]
    feature_names = []

    # Iterate over all attribute types (e.g., 'numeric', 'binary')
    for attr_type in feature_attrs.values():
        # Use list comprehension to extract feature names
        feature_names.extend([attr['name'] for attr in attr_type])

    return feature_names

#print coefficient of each feature for a model

def lr_coefficients(df, model, features="features"):
    coefficients = model.coefficients
    names = feature_names(df, features)
 
    weightsDF = pd.DataFrame(zip(names, coefficients), columns=['feature', 'weights'])
    # need to sort by absolute value
    sorted_list = weightsDF.reindex(weightsDF['weights'].abs().sort_values(ascending=False).index)
    return sorted_list
 
# print feature importance for decision tree, random forest and GBT (if one hot encoder is used)

def dt_featureImportance_ohe(df, model, features="features"):
  
  importance =model.featureImportances
 
  names=feature_names(df, features)
 
  weightsDF = pd.DataFrame(zip(names, importance), columns=['feature', 'importance'])
 
  sorted_list=weightsDF.sort_values('importance', ascending=False)
  return sorted_list

# check feature importance for the tree baded model without ohe
def dt_featureImportance(model, vecAssembler_index):
    featureImp = pd.DataFrame(
        list(zip(vecAssembler_index.getInputCols(), model.featureImportances)),
      columns=["feature", "importance"])
    return featureImp.sort_values(by="importance", ascending=False)

# Create a UDF to convert a vector to an array
 
from pyspark.sql.functions import udf, col, desc
from pyspark.sql.types import ArrayType, DoubleType
 
def to_array(col):
    def to_array_(v):
        return v.toArray().tolist()
    return udf(to_array_, ArrayType(DoubleType()))(col)
  
# create a function to return lift table
def lift (test, pred, cardinaility):
    # this dataframe is already sorted by probability of 1 in decending order
    res = pd.DataFrame(np.column_stack((test, pred)),
                       columns=['Target','PR_0', 'PR_1'])
    res['rank']=res['PR_0'].rank(method='first')
    res['Decile'] = pd.qcut(res['rank'], cardinaility, labels=False)+1
    crt = pd.crosstab(res.Decile, res.Target).reset_index()
    crt = crt.rename(columns= {'Target':'Np',0.0: 'Negatives', 1.0: 'Positives'})
 
    G = crt['Positives'].sum()
    B = crt['Negatives'].sum()
   
    avg_resp_rate = G/(G+B)
 
    crt['Response_Rate'] = round(crt['Positives']/(crt['Positives']+crt['Negatives']),2)
    crt['Lift'] = round((crt['Response_Rate']/avg_resp_rate),2)
    crt['rand_resp'] = 1/cardinaility
    crt['cmltv_p'] = round((crt['Positives']).cumsum(),2)
    crt['cmltv_p_perc'] = round(((crt['Positives']/G).cumsum())*100,1)
    crt['cmltv_n'] = round((crt['Negatives']).cumsum(),2)  
    crt['cmltv_n_perc'] = round(((crt['Negatives']/B).cumsum())*100,1)   
    crt['cmltv_rand_p_perc'] = (crt.rand_resp.cumsum())*100
    crt['Cumulated_Response_Rate'] = round(crt['cmltv_p']/(crt['cmltv_p']+crt['cmltv_n']),2)   
    crt['Cumulative_Lift'] = round(crt['Cumulated_Response_Rate']/avg_resp_rate,2)
    crt['KS']=round(crt['cmltv_p_perc']-crt['cmltv_rand_p_perc'],2)
    crt = crt.drop(['rand_resp','cmltv_p','cmltv_n','cmltv_p_perc', 'cmltv_n_perc','cmltv_rand_p_perc'], axis=1)
    
    print('average response rate: ' , avg_resp_rate)
    
    return crt

# display lift curve, gains curve and lift table
def display_lift_chart(predModel):
  
  from pyspark.sql.functions import desc
  import pandas as pd
  import matplotlib.pyplot as plt
  import scikitplot as skplt
  import numpy as np
  
  # create a pandas to store probability of 1 and probability of 0 for predicted label
  pd_prob=predModel.withColumn('prob_0', to_array(col('probability'))[0]).withColumn('prob_1', to_array(col('probability'))[1]).orderBy(desc('prob_1')).select('label','prob_0', 'prob_1').toPandas()
 
 
  Y_test=pd_prob['label']
  Y_test_pred=list(zip(pd_prob['prob_0'], pd_prob['prob_1']))
 
  # display cumulative Lift Curve
  skplt.metrics.plot_lift_curve(Y_test, Y_test_pred, figsize=(12, 8), title_fontsize=20, text_fontsize=18)
  
  print()
  
  # display cumulative gains curve 
  skplt.metrics.plot_cumulative_gain(Y_test, Y_test_pred, figsize=(12, 8), title_fontsize=20, text_fontsize=18)
  
  print()
  

def display_decile_table(predModel, cardinality=10):
  from pyspark.sql.functions import desc
  import pandas as pd
  import matplotlib.pyplot as plt
  import scikitplot as skplt
  import numpy as np
  
  # create a pandas to store probability of 1 and probability of 0 for predicted label
  pd_prob=predModel.withColumn('prob_0', to_array(col('probability'))[0]).withColumn('prob_1', to_array(col('probability'))[1]).orderBy(desc('prob_1')).select('label','prob_0', 'prob_1').toPandas()
 
 
  Y_test=pd_prob['label']
  Y_test_pred=list(zip(pd_prob['prob_0'], pd_prob['prob_1']))
 
  # display decile table
  modelLift=lift(Y_test, Y_test_pred,cardinality)
  return modelLift

    
def plot_roc(model):
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "r--")
    plt.plot(
        model.summary.roc.select("FPR").collect(),
        model.summary.roc.select("TPR").collect(),
    )
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.show()
    
def checkParameters(cv_model):
    avg_metrics = cv_model.avgMetrics
    for i, params in enumerate(cv_model.getEstimatorParamMaps()):
        print(f"Model {i+1} Params: ")
        for param, value in params.items():
            print(f"  {param}: {value}")
        print(f"  Average Metric: {avg_metrics[i]}\n")
        
def checkBestModel(best_model):
    # Print the parameters of the best model
    print("Best model parameters:")
    for param, value in best_model.extractParamMap().items():
        print(f"{param.name}: {value}")  
        
        
# pip install pyLDAvis --user   (if not already installed)

import numpy as np
import pyLDAvis

def visualize_lda(lda_model, vectorized_df, cv_model,
                  features_col='features', save_path=None):
    """
    Build a pyLDAvis visualization from a Spark ML LDA model.

    Parameters
    ----------
    lda_model     : fitted pyspark.ml.clustering.LDAModel
    vectorized_df : DataFrame with the CountVectorizer output column
    cv_model      : fitted CountVectorizerModel (for the vocabulary)
    features_col  : name of the feature vector column (default 'features')
    save_path     : optional path to save standalone HTML (e.g. 'lda_vis.html')

    Returns
    -------
    pyLDAvis PreparedData — display with pyLDAvis.display(vis)
    """
    # 1. Topic-term distributions (k × vocab)
    topic_term = lda_model.topicsMatrix().toArray().T
    topic_term_dists = topic_term / topic_term.sum(axis=1, keepdims=True)

    # 2. Doc-topic distributions (N × k)
    transformed = lda_model.transform(vectorized_df)
    doc_topic = np.array(
        transformed.select('topicDistribution')
                   .rdd.map(lambda r: r[0].toArray()).collect()
    )
    doc_topic_dists = doc_topic / doc_topic.sum(axis=1, keepdims=True)

    # 3. Doc lengths and term frequencies
    features = np.array(
        vectorized_df.select(features_col)
                     .rdd.map(lambda r: r[0].toArray()).collect()
    )
    doc_lengths = features.sum(axis=1).astype(int)
    term_frequency = features.sum(axis=0).astype(int)

    # 4. Prepare
    vis = pyLDAvis.prepare(
        topic_term_dists=topic_term_dists,
        doc_topic_dists=doc_topic_dists,
        doc_lengths=doc_lengths,
        vocab=cv_model.vocabulary,
        term_frequency=term_frequency,
        sort_topics=False
    )

    if save_path:
        pyLDAvis.save_html(vis, save_path)
        print(f"Saved to {save_path}")

    return vis