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
bd = pd.read_csv("db.csv", delimiter=";", nrows=25)
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
      <td>11177.78</td>
      <td>23</td>
      <td>748.27</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2633.90</td>
      <td>59</td>
      <td>196.82</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11026.95</td>
      <td>49</td>
      <td>804.84</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6739.66</td>
      <td>49</td>
      <td>462.12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9799.03</td>
      <td>48</td>
      <td>692.70</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2520.08</td>
      <td>28</td>
      <td>181.86</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1536.44</td>
      <td>22</td>
      <td>111.79</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9299.25</td>
      <td>41</td>
      <td>743.40</td>
    </tr>
    <tr>
      <th>8</th>
      <td>11487.56</td>
      <td>43</td>
      <td>819.04</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3389.47</td>
      <td>23</td>
      <td>213.46</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2876.39</td>
      <td>46</td>
      <td>231.55</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3181.83</td>
      <td>47</td>
      <td>246.92</td>
    </tr>
    <tr>
      <th>12</th>
      <td>11431.13</td>
      <td>44</td>
      <td>882.67</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3270.25</td>
      <td>58</td>
      <td>224.06</td>
    </tr>
    <tr>
      <th>14</th>
      <td>10872.05</td>
      <td>51</td>
      <td>701.65</td>
    </tr>
    <tr>
      <th>15</th>
      <td>10383.22</td>
      <td>29</td>
      <td>761.58</td>
    </tr>
    <tr>
      <th>16</th>
      <td>6113.69</td>
      <td>45</td>
      <td>479.88</td>
    </tr>
    <tr>
      <th>17</th>
      <td>8975.79</td>
      <td>37</td>
      <td>715.11</td>
    </tr>
    <tr>
      <th>18</th>
      <td>8837.50</td>
      <td>21</td>
      <td>650.09</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2896.92</td>
      <td>34</td>
      <td>232.70</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2417.69</td>
      <td>45</td>
      <td>168.13</td>
    </tr>
    <tr>
      <th>21</th>
      <td>10394.05</td>
      <td>35</td>
      <td>796.67</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2373.24</td>
      <td>21</td>
      <td>153.45</td>
    </tr>
    <tr>
      <th>23</th>
      <td>10739.36</td>
      <td>41</td>
      <td>818.07</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3444.25</td>
      <td>49</td>
      <td>214.71</td>
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
      <td>25.000000</td>
      <td>25.000000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6712.699200</td>
      <td>39.520000</td>
      <td>490.061600</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3785.781008</td>
      <td>11.597845</td>
      <td>280.392484</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1536.440000</td>
      <td>21.000000</td>
      <td>111.790000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2896.920000</td>
      <td>29.000000</td>
      <td>214.710000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6739.660000</td>
      <td>43.000000</td>
      <td>479.880000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>10394.050000</td>
      <td>48.000000</td>
      <td>748.270000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>11487.560000</td>
      <td>59.000000</td>
      <td>882.670000</td>
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


    array([748.27, 196.82, 804.84, 462.12, 692.7 , 181.86, 111.79, 743.4 ,
           819.04, 213.46, 231.55, 246.92, 882.67, 224.06, 701.65, 761.58,
           479.88, 715.11, 650.09, 232.7 , 168.13, 796.67, 153.45, 818.07,
           214.71])



```python
# Vamos ver a correlação entre as colunas
np.corrcoef(x, y)
```




    array([[1.        , 0.99122941],
           [0.99122941, 1.        ]])



# Precisamos fazer o reshape do X antes de enviar para o Sklearn


```python
x = x.reshape(-1, 1)
x
```




    array([[11177.78],
           [ 2633.9 ],
           [11026.95],
           [ 6739.66],
           [ 9799.03],
           [ 2520.08],
           [ 1536.44],
           [ 9299.25],
           [11487.56],
           [ 3389.47],
           [ 2876.39],
           [ 3181.83],
           [11431.13],
           [ 3270.25],
           [10872.05],
           [10383.22],
           [ 6113.69],
           [ 8975.79],
           [ 8837.5 ],
           [ 2896.92],
           [ 2417.69],
           [10394.05],
           [ 2373.24],
           [10739.36],
           [ 3444.25]])



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
grafico.add_scatter(x=x_teste.ravel(), y=previsao, name="Linha de regressão")
grafico.show()
```
![image](https://github.com/Welling7onSiqueira/regressao_linear_simples/assets/122923404/035684bd-2792-4245-8957-f50795b25ea2)

   </div>


