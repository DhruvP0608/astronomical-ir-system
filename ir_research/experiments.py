import json
from evaluation import ModelEvaluator

evaluator = ModelEvaluator()

results = evaluator.run()

with open("results.json","w") as f:
    json.dump(results,f,indent=4)

print("Results saved.")