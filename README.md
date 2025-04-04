Our goal for this project was to find a better way to predict, pre-snap, whether a team will pass the ball. The Run-Pass Oracle model, which we built on top of a LightGBM architecture, achieves performance that surpasses state-of-the-art literature with an .892 AUC, while still maintaining interpretability.

Much of this performance is driven by feature engineering, resulting in metrics such as Rectified Motion Value (RMV), Defensive Congestion Index (DCI), and Tempo Including Contextual Knowledge (TICK). These features heavily utilize the provided tracking data, and show a big boost over extant models, many of which barely eclipse 70% accuracy. Perhaps most notable of these models is Ben Baldwin's expected pass model; however, as the graphic below demonstrates, its predictions are still somewhat noisy

Baldwin's model includes important context like down and distance, and knows a little about team strength due to using Vegas odds. However, it doesn't incorporate the rich tracking data that our model utilizes. It also misses significantly on teams like the Chiefs and Bills, whose superstar QB's make them far more pass-happy than expected.

By incorporating tracking data, our model sees nearly a 10% improvement in accuracy over earlier literature, on both test and validation sets.
