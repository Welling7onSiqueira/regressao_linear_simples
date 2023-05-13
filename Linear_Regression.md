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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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


<div>                            <div id="fa4e79d4-fd79-4c8a-923f-ac3cf35169a4" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("fa4e79d4-fd79-4c8a-923f-ac3cf35169a4")) {                    Plotly.newPlot(                        "fa4e79d4-fd79-4c8a-923f-ac3cf35169a4",                        [{"hovertemplate":"x=%{x}<br>y=%{y}<extra></extra>","legendgroup":"","marker":{"color":"#636efa","symbol":"circle"},"mode":"markers","name":"","orientation":"v","showlegend":false,"x":[11177.78,2633.9,11026.95,6739.66,9799.03,2520.08,1536.44,9299.25,11487.56,3389.47,2876.39,3181.83,11431.13,3270.25,10872.05,10383.22,6113.69,8975.79,8837.5,2896.92,2417.69,10394.05,2373.24,10739.36,3444.25],"xaxis":"x","y":[748.27,196.82,804.84,462.12,692.7,181.86,111.79,743.4,819.04,213.46,231.55,246.92,882.67,224.06,701.65,761.58,479.88,715.11,650.09,232.7,168.13,796.67,153.45,818.07,214.71],"yaxis":"y","type":"scatter"},{"name":"Linha de regress\u00e3o","x":[2876.39,6113.69,2896.92,11177.78,10739.36,1536.44,9799.03,11487.56,8837.5],"y":[199.10927701898677,445.17730873077215,200.6697677646832,830.1001329087401,796.7757124154027,97.25932208981531,725.3009796876534,853.6465928796035,652.2148293364862],"type":"scatter"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"x"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"y"}},"legend":{"tracegroupgap":0},"margin":{"t":60}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('fa4e79d4-fd79-4c8a-923f-ac3cf35169a4');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>

