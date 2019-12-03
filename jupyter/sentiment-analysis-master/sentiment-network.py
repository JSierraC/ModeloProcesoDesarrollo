
# coding: utf-8

# ### Curación del dataset
# 
# En esta función podrémos analizar un review específico junto con su etiqueta, el dataset que estamos usando ya está preprocesado y contiene caracteres en minúsculas. Si trabajáramos a partir de datos en bruto, donde no sabíamos que todo estaba en minúsculas, nos gustaría agregar un paso mas aquí para convertirlo.

# In[1]:


def review_and_label(i):
    print(labels[i] + "\t:\t" + reviews[i][:80] + "...")


# Obtenémos el dataset de opiniones y lo convertimos a una lista línea por línea

# In[9]:


g = open('./reviews.txt','r')
reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()


# Obtenémos el dataset de etiquetas y lo convertimos a una lista línea por línea

# In[10]:


g = open('./labels.txt','r')
labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close()


# In[11]:


len(reviews)


# In[12]:


reviews[0]


# In[13]:


len(labels)


# In[14]:


labels[0]


# ### Desarrollamos una teoría predictiva

# In[19]:


print("Etiqueta \t : \t Opiniones\n")
review_and_label(213)
review_and_label(1286)
review_and_label(627)
review_and_label(234)


# Crearemos tres objetos de contador, uno para palabras de comentarios positivos, uno para palabras de comentarios negativos y uno para todas las palabras.

# In[20]:


from collections import Counter
import numpy as np


# Creamos tres objetos Count para almacenar conteos positivos, negativos y totales

# In[21]:


positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()


# Examinamos todas las revisiones. Para cada palabra en una revisión positiva, aumente el recuento de esa palabra tanto en su contador positivo como en el contador total de palabras; Del mismo modo, para cada palabra en una revisión negativa, aumente el recuento de esa palabra tanto en su contador negativo como en el contador total de palabras.
# 
# A lo largo de estos proyectos, debemos usar la división ('') para dividir una parte del texto (como una revisión) en palabras individuales. Si usamos split (), obtendremos resultados ligeramente diferentes.

# #### Pasamos por todas las palabras en todas las revisiones e incrementamos los recuentos en los objetos de contador apropiados

# In[22]:


counter = 0
for review in reviews:
    words = review.split(' ')
    if labels[counter] == 'POSITIVE':
        positive_counts.update(words)
    else:
        negative_counts.update(words)
    total_counts.update(words)
    counter += 1


# Enumeramos las palabras utilizadas en las revisiones positivas y negativas, respectivamente, ordenadas de la mayoría a las menos utilizadas.

# #### Examinamos los 20 primeros recuentos de las palabras más comunes en las revisiones positivas

# In[27]:


positive_counts.most_common()[:20]


# #### Examinamos los 20 primeros recuentos de las palabras más comunes en las revisiones negativas

# In[28]:


negative_counts.most_common()[:20]


# Como podémos ver, las palabras comunes como "the" aparecen muy a menudo en las revisiones positivas y negativas. En lugar de encontrar las palabras más comunes en las revisiones positivas o negativas, lo que realmente deseamos son las palabras que se encuentran en las revisiones positivas más a menudo que en las revisiones negativas, y viceversa. Para lograr esto, deberemos calcular las proporciones de uso de palabras entre las revisiones positivas y negativas.
# 
# ##### Sugerencia: 
# * la relación positiva-negativa para una palabra dada se puede calcular con positive_counts[word] / float(negative_counts[word]+1). Observamos el +1 en el denominador: eso asegura que no dividimos por cero las palabras que solo se ven en las revisiones positivas.

# In[29]:


pos_neg_ratios = Counter()
for word in total_counts.elements():
    if total_counts[word] >= 100:
        pos_neg_ratios[word] = positive_counts[word] / float(negative_counts[word] + 1)


# In[30]:


print("Pos-to-neg ratio para 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio para 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Pos-to-neg ratio para 'terrible' = {}".format(pos_neg_ratios["terrible"]))


# Si analizamos de cerca los valores que acabamos de calcular, verémos lo siguiente:
# 
# * Las palabras que esperaríamos ver más a menudo en las revisiones positivas, como "amazing", tienen una proporción mayor que 1. Cuanto más sesgada sea una palabra hacia positiva, más alejada de 1 será su relación positiva-negativa.
# * Las palabras que esperaríamos ver más a menudo en las revisiones negativas, como "terrible", tienen valores positivos que son menores a 1. Cuanto más sesgada es una palabra hacia negativa, más cercana a cero será su relación positiva a negativa.
# * Las palabras neutrales, que en realidad no transmiten ningún sentimiento porque esperaríamos verlas en todo tipo de reseñas, como "the", tienen valores muy cercanos a 1. Una palabra perfectamente neutral, una que se usó exactamente en el mismo número de revisiones positivas como críticas negativas - sería casi exactamente 1. El +1 que sugerimos agregar al denominador desvía ligeramente las palabras hacia negativo, pero no importará porque será un pequeño sesgo y luego ignoraremos las palabras que están demasiado cerca de neutral de todos modos.
# 
# Las proporciones nos dicen qué palabras se usan con más frecuencia en las revisiones positivas o positivas, pero los valores específicos que hemos calculado son un poco difíciles de trabajar. Una palabra muy positiva como "amazing" tiene un valor superior a 4, mientras que una palabra muy negativa como "terrible" tiene un valor de alrededor de 0.18.
# 
# Esos valores no son fáciles de comparar por un par de razones:
# 
# * En este momento, 1 se considera neutral, pero el valor absoluto de las raciones positivas a negativas de palabras muy positivas es mayor que el valor absoluto de las razones para las palabras muy negativas. Entonces no hay forma de comparar directamente dos números y ver si una palabra transmite la misma magnitud de sentimiento positivo ya que otra palabra transmite sentimiento negativo. Así que debemos centrar todos los valores en torno a netural de modo que el valor absoluto para neutro de la relación de posivo a negativo para una palabra indique cuánto sentimiento (positivo o negativo) transmite esa palabra.
# * Al comparar valores absolutos es más fácil hacer eso alrededor de cero que uno.
# 
# Para solucionar estos problemas, convertiremos todas nuestras proporciones en nuevos valores utilizando logaritmos.

# In[31]:


for word in pos_neg_ratios:
    ratio = pos_neg_ratios[word]
    pos_neg_ratios[word] = np.log(ratio)


# Examinamos las nuevas proporciones que hemos calculado para las mismas palabras de antes:

# In[32]:


print("Pos-to-neg ratio para 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio para 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Pos-to-neg ratio para 'terrible' = {}".format(pos_neg_ratios["terrible"]))


# Si todo funcionó, ahora deberíamos ver palabras neutrales con valores cercanos a cero. En este caso, "the" es casi cero pero ligeramente positivo, por lo que probablemente se usó en revisiones más positivas que en críticas negativas. Pero veamos la proporción "amazing" - está por encima de 1, mostrando que es claramente una palabra con sentimiento positivo. Y "terrible" tiene un puntaje similar, pero en la dirección opuesta, por lo que está por debajo de -1. Ahora está claro que ambas palabras están asociadas con sentimientos opuestos específicos.
# 
# ##### Analizamos palabras que se ven con mayor frecuencia en una revisión con una etiqueta "POSITIVO"

# In[34]:


pos_neg_ratios.most_common()[:20]


# ##### Analizamos palabras que se ven con mayor frecuencia en una revisión con una etiqueta "NEGATIVA"

# In[35]:


list(reversed(pos_neg_ratios.most_common()))[0:20]


# ### Creamos los datos de entrada / salida
# Creamos un conjunto denominado vocab que contenga cada palabra en el vocabulario.

# In[36]:


vocab = set(total_counts)
vocab_size = len(vocab)
print(vocab_size)


# Creamos una matriz numpy llamada layer_0 inicializada en 0. Asegúrate de crear layer_0 como una matriz bidimensional con columnas de 1 fila y vocab_size.

# In[37]:


layer_0 = np.zeros((1, vocab_size))


# In[38]:


layer_0.shape


# layer_0 contiene una entrada para cada palabra en el vocabulario. Necesitamos asegurarnos de que conocemos el índice de cada palabra.
# 
# ##### Creamos un diccionario de palabras en el vocabulario asignado a las posiciones de índice

# In[39]:


word2index = {}
for i,word in enumerate(vocab):
    word2index[word] = i


# Mostramos un mapa de palabras a índices

# In[49]:


c = 0
for w in word2index:
    if c < 20:
        print(w, end="")
        print(' : ', end="")
        print(word2index[w])
    c = c + 1


# #### Creamos la función update_input_layer. 
# Debe contar cuántas veces se usa cada palabra en la revisión dada, y luego almacenar esos conteos en los índices apropiados dentro de layer_0.

# In[50]:


def update_input_layer(review):
    global layer_0
    layer_0 *= 0
    for word in review.split(" "):
        layer_0[0][word2index[word]] += 1


# #### Probamos actualizando la capa de entrada con la primera opinión
# Los índices asignados pueden no ser los mismos que en la solución, pero con suerte verá algunos valores distintos de cero en layer_0.

# In[51]:


update_input_layer(reviews[0])
layer_0


# #### Creamos la unción  get_target_for_labels. 
# Debería devolver 0 o 1, dependiendo de si la etiqueta dada es NEGATIVA o POSITIVA, respectivamente.

# In[52]:


def get_target_for_label(label):
    if (label=="POSITIVE"):
        return 1
    else:
        return 0


# ##### Probamos con los primeros indices de las etiquetas y opiniones
# 
# Deberían imprimir 'POSITIVO' y 1, respectivamente.

# In[53]:


labels[0]


# In[54]:


get_target_for_label(labels[0])


# In[55]:


labels[1]


# In[56]:


get_target_for_label(labels[1])


# ### Construyendo una red neuronal

# Hemos incluido una clase llamada SentimentNetwork. 
# Implementamos todos los elementos marcados en el código. Estos deben hacer lo siguiente:
# 
# * Crear una red neuronal básica como las redes con una capa de entrada, una capa oculta y una capa de salida.
# * No agregamos una non-linearity en la capa oculta. Es decir, no usa una función de activación cuando calcule las salidas de la capa oculta.
# * Implementamos la función pre_process_data para crear el vocabulario de nuestras funciones de generación de datos de capacitación
# * Asegurar que se entrene sobre todo el corpus

# In[65]:


import time
import sys
import numpy as np
from collections import Counter


class SentimentNetwork:

    def __init__(self, reviews,labels,min_count = 10,polarity_cutoff = 0.1,hidden_nodes = 10, learning_rate = 0.1):
        """Creamos SentimenNetwork con la configuración dada
         Args:
             revisiones (lista) - Lista de revisiones usadas para entrenamiento
             labels (list) - Lista de etiquetas POSITIVAS / NEGATIVAS asociadas con las revisiones dadas
             min_count (int) - Las palabras solo deben agregarse al vocabulario
                              si ocurren más que esto muchas veces
             polarity_cutoff (float) - El valor absoluto de la palabra positiva a negativa
                                      la proporción debe ser al menos tan grande como para ser considerada.
             hidden_nodes (int) - Número de nodos para crear en la capa oculta
             learning_rate (float) - Tasa de aprendizaje para usar durante el entrenamiento
        
        """
        # Asignar una semilla a nuestro generador de números aleatorios para asegurarnos de obtener resultados reproducibles durante el desarrollo
        np.random.seed(1)

        # procesar las revisiones y sus etiquetas asociadas para que todo está listo para el entrenamiento
        self.pre_process_data(reviews, labels, polarity_cutoff, min_count)
        
        # Construye la red para tener la cantidad de nodos ocultos y la velocidad de aprendizaje que se pasaron a este inicializador. Haga la misma cantidad de nodos de entrada como hay palabras de vocabulario y crea un solo nodo de salida.
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels, polarity_cutoff, min_count):
        
        positive_counts = Counter()
        negative_counts = Counter()
        total_counts = Counter()

        for i in range(len(reviews)):
            if(labels[i] == 'POSITIVE'):
                for word in reviews[i].split(" "):
                    positive_counts[word] += 1
                    total_counts[word] += 1
            else:
                for word in reviews[i].split(" "):
                    negative_counts[word] += 1
                    total_counts[word] += 1

        pos_neg_ratios = Counter()

        for term,cnt in list(total_counts.most_common()):
            if(cnt >= 50):
                pos_neg_ratio = positive_counts[term] / float(negative_counts[term]+1)
                pos_neg_ratios[term] = pos_neg_ratio

        for word,ratio in pos_neg_ratios.most_common():
            if(ratio > 1):
                pos_neg_ratios[word] = np.log(ratio)
            else:
                pos_neg_ratios[word] = -np.log((1 / (ratio + 0.01)))

        # poblar review_vocab con todas las palabras en las revisiones dadas
        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                if(total_counts[word] > min_count):
                    if(word in pos_neg_ratios.keys()):
                        if((pos_neg_ratios[word] >= polarity_cutoff) or (pos_neg_ratios[word] <= -polarity_cutoff)):
                            review_vocab.add(word)
                    else:
                        review_vocab.add(word)

        # Convertir el conjunto de vocabulario en una lista para que podamos acceder a las palabras a través de índices
        self.review_vocab = list(review_vocab)
        
        # poblar etiqueta_vocab con todas las palabras en las etiquetas dadas.
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
        
        # Convertir el conjunto de vocabulario de la etiqueta en una lista para que podamos acceder a las etiquetas a través de índices
        self.label_vocab = list(label_vocab)
        
        # Almacenar los tamaños de los vocabularios de revisión y etiqueta.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        # Crear un diccionario de palabras en el vocabulario asignado a las posiciones de índice
        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i
        
        # Crear un diccionario de etiquetas mapeadas a posiciones de índice
        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i

    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Establecer el número de nodos en las capas de entrada, ocultas y de salida.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Almacenar la tasa de aprendizaje
        self.learning_rate = learning_rate

        # Inicializar los pesos Estos son los pesos entre la capa de entrada y la capa oculta.
        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))

        # Estos son los pesos entre la capa oculta y la capa de salida.
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))
        
        # La capa de entrada, una matriz bidimensional con forma 1 x hidden_nodes
        self.layer_1 = np.zeros((1,hidden_nodes))
    
    def get_target_for_label(self,label):
        if(label == 'POSITIVE'):
            return 1
        else:
            return 0
        
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_output_2_derivative(self,output):
        return output * (1 - output)
    
    def train(self, training_reviews_raw, training_labels):

        ## Preprocesamiento de las evaluaciones de capacitación para que podamos tratar directamente con los índices de entradas distintas de cero
        training_reviews = list()
        for review in training_reviews_raw:
            indices = set()
            for word in review.split(" "):
                if(word in self.word2index.keys()):
                    indices.add(self.word2index[word])
            training_reviews.append(list(indices))

        # asegúrate de que tenemos un número coincidente de reseñas y etiquetas
        assert(len(training_reviews) == len(training_labels))
        
        # Realizar un seguimiento de las predicciones correctas para mostrar la precisión durante el entrenamiento
        correct_so_far = 0

        # Recuerda cuando comenzamos a imprimir las estadísticas de tiempo
        start = time.time()
        
        # recorrer todas las evaluaciones dadas y ejecutar un pase hacia adelante y hacia atrás, actualización de pesos para cada artículo
        for i in range(len(training_reviews)):
            
            # Obtener la siguiente revisión y su etiqueta correcta
            review = training_reviews[i]
            label = training_labels[i]
            self.layer_1 *= 0
            for index in review:
                self.layer_1 += self.weights_0_1[index]

            # Output layer
            layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))            
            
            # Output error
            layer_2_error = layer_2 - self.get_target_for_label(label) # El error de la capa de salida es la diferencia entre el objetivo deseado y la salida real.
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)

            # Backpropagated error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T) # errores propagados a la capa oculta
            layer_1_delta = layer_1_error # gradientes de capas ocultas, sin falta de linealidad, es el mismo que el error

            # Actualiza los pesos
            self.weights_1_2 -= self.layer_1.T.dot(layer_2_delta) * self.learning_rate # actualizar pesos ocultos a salida con paso de descenso de degradado
            
            for index in review:
                self.weights_0_1[index] -= layer_1_delta[0] * self.learning_rate # actualizar pesos de entrada a ocultos con paso de descenso de gradiente

            # Manten un registro de las predicciones correctas.
            if(layer_2 >= 0.5 and label == 'POSITIVE'):
                correct_so_far += 1
            elif(layer_2 < 0.5 and label == 'NEGATIVE'):
                correct_so_far += 1
            
            # Para depuración, imprime nuestra precisión y velocidad de predicción a lo largo del proceso de capacitación.
            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4]                              + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5]                              + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1)                              + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
    
    def test(self, testing_reviews, testing_labels):
        """
        Intenta predecir las etiquetas para las evaluaciones de prueba dadas,
         y usa test_labels para calcular la precisión de esas predicciones.
        """
        
        # realizar un seguimiento de la cantidad de predicciones correctas que hacemos
        correct = 0

        # vamos a cronometrar cuántas predicciones por segundo hacemos
        start = time.time()

        # Pasa por cada una de las revisiones dadas y ejecuta la llamada para predecir su etiqueta.
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            
            # Para depuración, imprima nuestra precisión y velocidad de predicción durante todo el proceso de predicción.

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4]                              + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5]                              + " #Correct:" + str(correct) + " #Tested:" + str(i+1)                              + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def run(self, review):
        """
        Devuelve una predicción POSITIVA o NEGATIVA para la revisión dada.
        """
        ## Hidden layer
        self.layer_1 *= 0
        unique_indices = set()
        for word in review.lower().split(" "):
            if word in self.word2index.keys():
                unique_indices.add(self.word2index[word])
        for index in unique_indices:
            self.layer_1 += self.weights_0_1[index]
        
        ## Output layer
        layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))
         
        # Devuelve POSITIVO para valores superiores a mayor que o igual a 0.5 en la capa de salida; devuelve NEGATIVO para otros valores
        if(layer_2[0] >= 0.5):
            return "POSITIVE"
        else:
            return "NEGATIVE"


# #### entrenamos la red con un pequeño corte de polaridad.

# In[66]:


mlp_full = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=0,polarity_cutoff=0,learning_rate=0.01)


# In[67]:


mlp_full.train(reviews[:-1000],labels[:-1000])


# Nuestra Red Neural está lista para decir si una nueva opinión es "POSITIVA" o "NEGATIVA", probemos una negativa

# In[68]:


mlp_full.run("This movie was very bad and I did not like it")


# In[69]:


mlp_full.run("I loved this movie, it was the best ever")

