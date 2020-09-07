<p align="center">
  <img width="1272" alt="image" src="https://user-images.githubusercontent.com/24773116/75319337-d1c39b80-58af-11ea-83f1-6c0d2a0bbad1.png">
  <img width="80" alt="image" src="https://user-images.githubusercontent.com/24773116/75319298-bc4e7180-58af-11ea-8df2-eac383cdad73.png">
</p>

<p align="center">
  <code>Batcar</code> is a <b>tool for evaluating model through time</b> in order to <b>narrow the gap between research and deployment</b>. Batcar evaluates & updates model through time just like in real situation
</p>

## 🗨️ Usage
- For business
  - Persuade need of continual learning by showing result when it's applied/not applied
  - Investigate amount of data needed for training by monitoring model performance
- For analysis
  - Investigate distribution changes (trend / seasonality) by monitoring model performance
- For research
  - Compare different update triggers
  - Compare different model update methods


## ⚡️ Quickstart
```python
import batcar
batcar = BatCar()
batcar.drive(x, y, model)

model_archive = batcar.history['models']
history_pred = batcar.history['pred']
history_eval = batcar.history['eval']
```
- x, y
  - nd.ndarray or pd.DataFrame, pd.Series
  - supports pd.RangeIndex, pd.Int64Index, pd.DatetimeIndex
- model
  - should be able to train with .fit()
  - should be able to predict with .predict()

## 🕐 Time traveler
```python

```

## 🔫 Update trigger
```python

```

## 🔧 Model updater
```python

```

## 🛠 Installation
```bash
$ git clone https://github.com/makinarocks/batcar.git
...
```

## 🔩 Development
```bash
# Update document when code changes
pip3 install pdoc3
pdoc3 --html batcar --force
mv html/batcar docs
rm -r html
```

## 🧠 Philosophy
content
## 🔥 Features
-
## 🔗 Useful links
- [Documentation](https://github.com/makinarocks/project-solar/tree/develop-add-batcar/batcar/docs)
