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


<div>                            <div id="5d59aeaf-bad9-4c78-9266-22e7672f19a5" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("5d59aeaf-bad9-4c78-9266-22e7672f19a5")) {                    Plotly.newPlot(                        "5d59aeaf-bad9-4c78-9266-22e7672f19a5",                        [{"hovertemplate":"x=%{x}<br>y=%{y}<extra></extra>","legendgroup":"","marker":{"color":"#636efa","symbol":"circle"},"mode":"markers","name":"","orientation":"v","showlegend":false,"x":[2337.27,2561.7,11652.21,8274.59,9952.98,6637.47,10387.27,5513.91,7953.64,10023.61,2621.11,10853.2,7640.32,9726.02,6311.98,3175.73,10414.09,3754.11,10000.25,3250.22],"xaxis":"x","y":[119.2,125.52,268.0,372.36,278.68,365.06,529.75,281.21,469.26,431.02,70.77,586.07,435.5,291.78,164.11,196.9,208.28,67.57,590.01,61.75],"yaxis":"y","type":"scatter"},{"x":[10853.2,2337.27,3175.73,6311.98,10023.61,10387.27,2621.11],"y":[393.88937187320306,139.81668805079403,164.8321335133257,258.40211995497447,369.1385627692963,379.98835512302134,148.28505152679168],"type":"scatter"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"x"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"y"}},"legend":{"tracegroupgap":0},"margin":{"t":60}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('5d59aeaf-bad9-4c78-9266-22e7672f19a5');
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

