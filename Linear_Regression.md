# Regressão Linear simples

A base foi gerada aleatória, e esse é apenas um exemplo simples da implementação.


```python
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
```

# Carregar a base de dados


```python
bd = pd.read_csv("db.csv", delimiter=";", nrows=20)
bd
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Salario</th>
      <th>Idade</th>
      <th>Custo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2337.27</td>
      <td>51</td>
      <td>119.20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2561.70</td>
      <td>49</td>
      <td>125.52</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11652.21</td>
      <td>23</td>
      <td>268.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8274.59</td>
      <td>45</td>
      <td>372.36</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9952.98</td>
      <td>28</td>
      <td>278.68</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6637.47</td>
      <td>55</td>
      <td>365.06</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10387.27</td>
      <td>51</td>
      <td>529.75</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5513.91</td>
      <td>51</td>
      <td>281.21</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7953.64</td>
      <td>59</td>
      <td>469.26</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10023.61</td>
      <td>43</td>
      <td>431.02</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2621.11</td>
      <td>27</td>
      <td>70.77</td>
    </tr>
    <tr>
      <th>11</th>
      <td>10853.20</td>
      <td>54</td>
      <td>586.07</td>
    </tr>
    <tr>
      <th>12</th>
      <td>7640.32</td>
      <td>57</td>
      <td>435.50</td>
    </tr>
    <tr>
      <th>13</th>
      <td>9726.02</td>
      <td>30</td>
      <td>291.78</td>
    </tr>
    <tr>
      <th>14</th>
      <td>6311.98</td>
      <td>26</td>
      <td>164.11</td>
    </tr>
    <tr>
      <th>15</th>
      <td>3175.73</td>
      <td>62</td>
      <td>196.90</td>
    </tr>
    <tr>
      <th>16</th>
      <td>10414.09</td>
      <td>20</td>
      <td>208.28</td>
    </tr>
    <tr>
      <th>17</th>
      <td>3754.11</td>
      <td>18</td>
      <td>67.57</td>
    </tr>
    <tr>
      <th>18</th>
      <td>10000.25</td>
      <td>59</td>
      <td>590.01</td>
    </tr>
    <tr>
      <th>19</th>
      <td>3250.22</td>
      <td>19</td>
      <td>61.75</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Observar os dados
# Outras analises podem ser feitas
bd.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Salario</th>
      <th>Idade</th>
      <th>Custo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20.000000</td>
      <td>20.000000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7152.084000</td>
      <td>41.350000</td>
      <td>295.640000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3236.676572</td>
      <td>15.526802</td>
      <td>171.351199</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2337.270000</td>
      <td>18.000000</td>
      <td>61.750000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3628.137500</td>
      <td>26.750000</td>
      <td>154.462500</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7796.980000</td>
      <td>47.000000</td>
      <td>279.945000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>10006.090000</td>
      <td>54.250000</td>
      <td>432.140000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>11652.210000</td>
      <td>62.000000</td>
      <td>590.010000</td>
    </tr>
  </tbody>
</table>
</div>



# Agora é separar a base em X e Y


```python
x = bd.iloc[:,0].values # Nesse caso, estou levando em consideração só a coluna de salario
y = bd.iloc[:,2].values # Aqui a coluna que quero prever, no caso a de custo
```


```python
# Visualizando x e y
# display(x)
display(y)
```


    array([119.2 , 125.52, 268.  , 372.36, 278.68, 365.06, 529.75, 281.21,
           469.26, 431.02,  70.77, 586.07, 435.5 , 291.78, 164.11, 196.9 ,
           208.28,  67.57, 590.01,  61.75])



```python
# Vamos ver a correlação entre as colunas
np.corrcoef(x, y)
```




    array([[1.        , 0.73206591],
           [0.73206591, 1.        ]])



# Precisamos fazer o reshape do X antes de enviar para o Sklearn


```python
x = x.reshape(-1, 1)
x
```




    array([[ 2337.27],
           [ 2561.7 ],
           [11652.21],
           [ 8274.59],
           [ 9952.98],
           [ 6637.47],
           [10387.27],
           [ 5513.91],
           [ 7953.64],
           [10023.61],
           [ 2621.11],
           [10853.2 ],
           [ 7640.32],
           [ 9726.02],
           [ 6311.98],
           [ 3175.73],
           [10414.09],
           [ 3754.11],
           [10000.25],
           [ 3250.22]])



# Vamos separar a base de treino e teste


```python
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.33, random_state=30)
```

# Vamos treinar o modelo


```python
regressor = LinearRegression()
regressor.fit(x_treino, y_treino)

regressor.score(x_teste, y_teste)
previsao = regressor.predict(x_teste)
```

# Vamos observar o modelo através de um gráfico



```python
grafico = px.scatter(x=x.ravel(), y=y)
grafico.add_scatter(x=x_teste.ravel(), y=previsao)
grafico.show()
```
![image](https://github.com/Welling7onSiqueira/regressao_linear_simples/assets/122923404/855c5891-500d-4602-addd-572489ec9560)

```python
# Aqui como os valores da tabela não estão com uma Correlação forte, o grafico consegue mostrar essa diferença.
# Teste com outras bases de dados, para visualizar a mudança de acordo com os dados fornecidos 
```

  </div>

