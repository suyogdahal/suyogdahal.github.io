---
title : 'How a Python Decorator Crashed My Flask App: Lessons Learned'
## alternate title Python Decorators: How One Crashed My Flask App and the Lessons Learned
date : 2024-07-01T11:07:28+05:45
draft : true
tags : ["python"]
---

Python decorators are wonderful. They let you modify or extend the behavior of functions or methods without permanently modifying their source code. I've used them in several places in my code to add features to my functions. But recently, I came across a weird, or rather interesting, bug that I felt was worth sharing.
 
So I had a Flask API that was responsible for ML inference. Here is what the rough structure of the project looked like:

```
project/
├── app/
│   ├── __init__.py
│   ├── app.py
│   ├── decorators.py
│   ├── ...
```

`app/app.py`
```python
# all other imports
....
from .decorators import timeit

@app.route('/v1/predict/', methods=['POST'])
@timeit
def predict():
    # do something
```

So as you can see above, I had a fairly simple decorator for timing the execution of an endpoint. Here is what a rough implementation of the decorator looked like: 

`app/decorators.py`

```python
import time

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper
```

For those Python wizards out there who can immediately identify the issue by seeing the snippets above, feel free to skip the rest of this tutorial (you guys have my respect). For those who do not see any issue with the above snippets, please read along. You deserve to know the issue! :)

