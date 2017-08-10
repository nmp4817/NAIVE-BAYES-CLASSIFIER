import time
start_time = time.time()
import numpy
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter


mytokenizer = RegexpTokenizer(r'[a-zA-Z0-9//]+')
stemmer = PorterStemmer()
sortedstopwords = sorted(stopwords.words('english'))      	        

def readfiles(s):
    return pd.read_csv("C:/Users/NabilPatel/Desktop/DataMining/Assignment-2/"+s, encoding="ISO-8859-1")

def convert_to_dictioinary(series1,series2):
    return dict([(i,a) for i,a in zip(series1,series2)])

def preprocessing(dict_xyz):
    for k,v in dict_xyz.items():
        k1 = int(k)
        if isinstance(v,str):
            v1 = str(u' '.join((v,'')).encode('utf-8'))
        else:
            v1 = str(v)
	    
        v_tokens = mytokenizer.tokenize(v1)
        v_lowertokens = [token.lower() for token in v_tokens]
        v_filteredtokens = [stemmer.stem(token) for token in v_lowertokens if not token in sortedstopwords]
        dict_xyz[k1] = v_filteredtokens
        
    return dict_xyz
	
def create_features(dict_query_after_preprocessing,attribute,dict_train_or_test_product):
    
    feature = {}
    count = 0
    count1 = 0
    
    for k,v in dict_query_after_preprocessing.items():
        
        product_uid = dict_train_or_test_product.get(k)
        attribute_title = attribute.get(product_uid)
        flag = 0
        
        for i in range(len(v)):
            for j in range(len(attribute_title)):
                if v[i] == attribute_title[j]:
                    count += 1
                    feature[k] = "yes"
                    flag = 1
                    break
            if flag == 1:
                break
        if flag == 0:
            count1 += 1
            feature[k] = "no"
            
    return feature
	
def naive_bayes_model(df_train_relevance,dict_train_product,dict_train_relevance,attribute_train_product_title,attribute_train_product_description,attribute_train_brand,attribute_train_material):
    
    dict_probability_title = [None] * 2
    dict_probability_description = [None] * 2
    dict_probability_brand = [None] * 2
    dict_probability_material = [None] * 2
    result = [None] * 16
    
    list_train_relevance = df_train_relevance.tolist()
    dict_class_relevance_count = Counter(list_train_relevance)
    dict_class_probability = probability_of_class(dict_class_relevance_count,dict_train_product)
    
    dict_probability_title[1] = probability_of_attributes(attribute_train_product_title,dict_train_relevance,dict_class_relevance_count,'yes')
    dict_probability_title[0] = probability_of_attributes(attribute_train_product_title,dict_train_relevance,dict_class_relevance_count,'no')
    
    dict_probability_description[1] = probability_of_attributes(attribute_train_product_description,dict_train_relevance,dict_class_relevance_count,'yes')
    dict_probability_description[0] = probability_of_attributes(attribute_train_product_description,dict_train_relevance,dict_class_relevance_count,'no')
	
    dict_probability_brand[1] = probability_of_attributes(attribute_train_brand,dict_train_relevance,dict_class_relevance_count,'yes')
    dict_probability_brand[0] = probability_of_attributes(attribute_train_brand,dict_train_relevance,dict_class_relevance_count,'no')

    dict_probability_material[1] = probability_of_attributes(attribute_train_material,dict_train_relevance,dict_class_relevance_count,'yes')
    dict_probability_material[0] = probability_of_attributes(attribute_train_material,dict_train_relevance,dict_class_relevance_count,'no')

    for x in range (0,16):
        index_for_attributes = [int(i) for i in bin(x)[2:].zfill(4)]
        result[x] = naive_bayes_calculation(dict_probability_title[index_for_attributes[0]],dict_probability_description[index_for_attributes[1]],dict_probability_brand[index_for_attributes[2]],dict_probability_material[index_for_attributes[3]],dict_class_probability)
	
    return result

def probability_of_attributes(attribute_train_product_title_or_description_or_brand_or_material,dict_train_relevance,dict_class_relevance_count,value):
    
    dict_count_value = {}
    dict_probability_value = {}
    
    for k,v in dict_class_relevance_count.items():
        dict_count_value[k] = 0
        
    for token in [k for k,v in attribute_train_product_title_or_description_or_brand_or_material.items() if v == value]:
        dict_count_value[dict_train_relevance.get(token)] += 1
        
    for k,v in dict_count_value.items():
        dict_probability_value[k] = float(v)/dict_class_relevance_count[k]
	
    return dict_probability_value
	
def probability_of_class(dict_class_relevance_count,dict_train_product):
    dict_class_probability = {}
    
    for k,v in dict_class_relevance_count.items():
        dict_class_probability[k] = float(v)/len(dict_train_product)
    
    return dict_class_probability
    
def naive_bayes_calculation(dict_value_title,dict_value_description,dict_value_brand,dict_value_material,dict_class_probability):
    
    dict_value_value_value_value_class = {}
    
    for k,v in dict_value_title.items():
        dict_value_value_value_value_class[k] = v * dict_value_description[k] * dict_value_brand[k] * dict_value_material[k] * dict_class_probability[k]
    
    return [k for k,v in dict_value_value_value_value_class.items() if v == max(dict_value_value_value_value_class.values())]
    
def calculate_result(model,attribute_test_product_title,attribute_test_product_description,attribute_test_brand,attribute_test_material):
    dict_result = {}
    
    for k,v in attribute_test_product_title.items():
        if attribute_test_product_title[k] == 'yes':
            s = '1'
        else:
            s = '0'
        
        if attribute_test_product_description[k] == 'yes':
            s = s+'1'
        else:
            s = s+'0'
        
        if attribute_test_brand[k] == 'yes':
            s = s+'1'
        else:
            s = s+'0'

        if attribute_test_material[k] == 'yes':
            s = s+'1'
        else:
            s = s+'0'
			
        dict_result[k] = model[int(s,2)][0]
	    
    return pd.DataFrame(list(dict_result.items()),columns=['id','relevance'])
	
if __name__ == "__main__":
    df_test = readfiles("test.csv")
    df_train = readfiles("train.csv")
    df_attributes = readfiles("attributes.csv")
    df_description = readfiles("product_descriptions.csv")
    
    df_brand = df_attributes[df_attributes.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
    df_material = df_attributes[df_attributes.name == "Material"][["product_uid", "value"]].rename(columns={"value": "material"})
    
    dict_product_test = convert_to_dictioinary(df_test.product_uid,df_test.product_title)
    dict_product_train = convert_to_dictioinary(df_train.product_uid,df_train.product_title)
    dict_all_product_b4_preprocessing = {**dict_product_test,**dict_product_train}
    
    dict_test_product = convert_to_dictioinary(df_test.id,df_test.product_uid)
    dict_train_product = convert_to_dictioinary(df_train.id,df_train.product_uid)
    
    dict_test_query_b4_preprocessing = convert_to_dictioinary(df_test.id,df_test.search_term)
    dict_train_query_b4_preprocessing = convert_to_dictioinary(df_train.id,df_train.search_term)
    
    dict_train_relevance = convert_to_dictioinary(df_train.id,df_train.relevance)
    
    dict_product_description_b4_preprocessing = convert_to_dictioinary(df_description.product_uid,df_description.product_description)
    
    dict_brand_b4_preprocessing = convert_to_dictioinary(df_brand.product_uid,df_brand.brand)
    
    dict_material_b4_preprocessing = convert_to_dictioinary(df_material.product_uid,df_material.material)
    
    dict_all_product_after_preprocessing = preprocessing(dict_all_product_b4_preprocessing)
    
    dict_test_query_after_preprocessing = preprocessing(dict_test_query_b4_preprocessing)
    
    dict_train_query_after_preprocessing = preprocessing(dict_train_query_b4_preprocessing)
    
    dict_product_description_after_preprocessing = preprocessing(dict_product_description_b4_preprocessing)
    
    dict_brand_after_preprocessing = preprocessing(dict_brand_b4_preprocessing)
    
    dict_material_after_preprocessing = preprocessing(dict_material_b4_preprocessing)
    
    for k,v in dict_all_product_after_preprocessing.items():
        if k not in dict_brand_after_preprocessing:
            dict_brand_after_preprocessing[k] = 'unknown'

    for k,v in dict_all_product_after_preprocessing.items():
        if k not in dict_material_after_preprocessing:
            dict_material_after_preprocessing[k] = 'unknown'
    
    attribute_train_product_title = create_features(dict_train_query_after_preprocessing,dict_all_product_after_preprocessing,dict_train_product)
    attribute_train_product_description = create_features(dict_train_query_after_preprocessing,dict_product_description_after_preprocessing,dict_train_product)
    attribute_train_brand = create_features(dict_train_query_after_preprocessing,dict_brand_after_preprocessing,dict_train_product)
    attribute_train_material = create_features(dict_train_query_after_preprocessing,dict_material_after_preprocessing,dict_train_product)
    
    model = naive_bayes_model(df_train.relevance,dict_train_product,dict_train_relevance,attribute_train_product_title,attribute_train_product_description,attribute_train_brand,attribute_train_material)
    
    attribute_test_product_title = create_features(dict_test_query_after_preprocessing,dict_all_product_after_preprocessing,dict_test_product)
    attribute_test_product_description = create_features(dict_test_query_after_preprocessing,dict_product_description_after_preprocessing,dict_test_product)
    attribute_test_brand = create_features(dict_test_query_after_preprocessing,dict_brand_after_preprocessing,dict_test_product)
    attribute_test_material = create_features(dict_test_query_after_preprocessing,dict_material_after_preprocessing,dict_test_product)
    
    calculate_result(model,attribute_test_product_title,attribute_test_product_description,attribute_test_brand,attribute_test_material).to_csv('submission.csv',index=False)
    
    print("--- %s seconds ---" % (time.time() - start_time))

   