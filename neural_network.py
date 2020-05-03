def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('G3')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

def demo(feature_column,example_batch):
  feature_layer = layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())

def main():
    feature_columns = []
    
    batch_size = 32;

    dataframe = pd.read_csv('/Volumes/macOS/OneDrive/workspace/Python/Student_Performance/databases/student-por.csv',delimiter=';');
    train, test = train_test_split(dataframe, test_size=0.4)
    train, val = train_test_split(train, test_size=0.4)
    
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
    
    example_batch = next(iter(train_ds))[0]
    
    # Variáveis Binárias
    school = feature_column.categorical_column_with_vocabulary_list(
      'school', ['GP','MS']
    ); school_result = feature_column.indicator_column(school); feature_columns.append(school_result);

    sex = feature_column.categorical_column_with_vocabulary_list(
      'sex', ['M','F']
    ); sex_result = feature_column.indicator_column(sex); feature_columns.append(sex_result);

    address = feature_column.categorical_column_with_vocabulary_list(
      'address',['U','R']
    ); address_result = feature_column.indicator_column(address); feature_columns.append(address_result);

    famsize = feature_column.categorical_column_with_vocabulary_list(
      'famsize',['LE3','GT3']
    ); famsize_result = feature_column.indicator_column(famsize); feature_columns.append(famsize_result);

    status = feature_column.categorical_column_with_vocabulary_list(
      'Pstatus',['T','A']
    ); status_result = feature_column.indicator_column(status); feature_columns.append(status_result);

    school_sup = feature_column.categorical_column_with_vocabulary_list(
      'schoolsup',['yes','no']
    ); school_sup_result = feature_column.indicator_column(school_sup); feature_columns.append(school_sup_result);

    famsup = feature_column.categorical_column_with_vocabulary_list(
      'famsup',['yes','no']
    ); famsup_result = feature_column.indicator_column(famsup); feature_columns.append(famsup_result);

    paid = feature_column.categorical_column_with_vocabulary_list(
      'paid',['yes','no']
    ); paid_result = feature_column.indicator_column(paid); feature_columns.append(paid_result)

    activities = feature_column.categorical_column_with_vocabulary_list(
      'activities',['yes','no']
    ); activities_result = feature_column.indicator_column(activities); feature_columns.append(activities_result);

    nursery = feature_column.categorical_column_with_vocabulary_list(
      'nursery',['yes','no']
    ); nursery_result = feature_column.indicator_column(nursery); feature_columns.append(nursery_result);

    higher = feature_column.categorical_column_with_vocabulary_list(
      'higher',['yes','no']
    ); higher_result = feature_column.indicator_column(higher); feature_columns.append(higher_result);

    internet = feature_column.categorical_column_with_vocabulary_list(
      'internet',['yes','no']
    ); internet_result = feature_column.indicator_column(internet); feature_columns.append(internet_result);

    romantic = feature_column.categorical_column_with_vocabulary_list(
      'romantic',['yes','no']
    ); romantic_result = feature_column.indicator_column(romantic); feature_columns.append(romantic_result);
  #
    
  # Variáveis Categóricas
    mother_job = feature_column.categorical_column_with_vocabulary_list(
      'Mjob',['teacher','health','services','at_home','other']
    ); mother_job_result = feature_column.indicator_column(mother_job); feature_columns.append(mother_job_result);

    father_job = feature_column.categorical_column_with_vocabulary_list(
      'Fjob',['teacher','health','services','at_home','other']
    ); father_job_result = feature_column.indicator_column(father_job); feature_columns.append(father_job_result);

    reason = feature_column.categorical_column_with_vocabulary_list(
      'reason',['home','reputation','course','other']
    ); reason_result = feature_column.indicator_column(reason); feature_columns.append(reason_result);

    guardian = feature_column.categorical_column_with_vocabulary_list(
      'guardian',['mother','father','other']
    ); guardian_result = feature_column.indicator_column(guardian); feature_columns.append(guardian_result);

  # Variáveis Númericas
    for header in ['age','Medu','Fedu','traveltime','studytime','failures','famrel','freetime','goout','Dalc','Walc','health','absences']:
      feature_columns.append(feature_column.numeric_column(header))

    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    
    model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(128,activation='relu'),
        layers.Dense(128,activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanAbsoluteError(),
                  metrics=['accuracy'])

    model.fit(train_ds,
          validation_data=val_ds,
          epochs=200,
          use_multiprocessing=True
      )

    loss, accuracy= model.evaluate(test_ds)

    print("Acurácia --> {:.4f}".format(accuracy*100))



try:
    import numpy as np
    import pandas as pd
    
    import csv

    import tensorflow as tf

    from tensorflow import feature_column
    from tensorflow.python.keras import layers
    from sklearn import metrics;
    from sklearn.model_selection import train_test_split    
    main()

except ImportError as error:
    print(error)
 

