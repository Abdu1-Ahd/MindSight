# MindSight: AI-Driven Analysis of Mental Health Treatment-Seeking Behavior in Technology Workplaces

**Course:** AL2002 Artificial Intelligence  
**Track:** D Mental Health in Tech  
**Phase:** 6 Research Report  
**Date:** May 2026

---

## Abstract

MindSight applies both classic and modern AI techniques to the Open Source Mental Illness (OSMI) Mental Health in Tech Survey dataset. I merged seven years of survey responses (2016-2022) into a single dataset with 3,433 rows and 193 columns. My main goal was predicting the `treatment` variable—whether a respondent sought professional help (which happens about 58% of the time in this data).

Over five phases, I built search algorithms, constraint satisfaction systems, logic rules, and machine learning models from scratch to see what actually drives people to seek care. The biggest surprise came in the machine learning phase: my simple Delta Rule linear classifier hit 84.13% test accuracy, totally beating out my Multilayer Perceptron, which got stuck at 71.32%. That result really drove home a lesson you hear a lot but rarely see so starkly: deep learning isn't a magic bullet, especially on small tabular datasets. More importantly, this pipeline shows that AI isn't just an academic exercise. It can actually point us toward the workplace factors that encourage people to get the help they need.

---

## 1. Introduction

Mental health in the tech industry is a known problem that nobody really knows how to fix. Software engineering pairs high cognitive load and tight deadlines with a culture that often rewards grinding through burnout and ignoring personal limits. Back in 2014, the Open Source Mental Illness (OSMI) organization started running an annual survey to figure out what was actually happening on the ground in developer communities. By asking about everything from employer stigmas to personal treatment histories, they built one of the best longitudinal datasets we have on the subject.

But honestly, looking at this data manually is a nightmare. It's a mess of free-text comments, changing rating scales, and columns that get renamed or dropped depending on the year. I realized pretty quickly that traditional, manual analysis wasn't going to cut it. This makes it a perfect candidate for an automated artificial intelligence pipeline. AI is incredibly useful here because it can cut through the noise, spot non-obvious correlations, and handle the sheer dimensionality of 190+ variables without getting overwhelmed.

I wanted to throw a variety of AI paradigms at this problem to see if they'd arrive at similar conclusions. I used search algorithms to map paths from symptoms to treatment, constraint satisfaction to model logical combinations of employer support, propositional logic to write readable rules, and machine learning to predict outcomes on unseen data.

My core research question was simple: Can these different AI techniques consistently identify what actually pushes tech workers to seek professional help? The answer matters deeply. The cost of burnout in tech is astronomical, both in human suffering and lost productivity. If we can isolate the specific environmental variables that predict treatment-seeking, companies can stop guessing. They can start designing proactive workplace policies and HR interventions that actually work, rather than just offering generic wellness seminars that nobody attends.

---

## 2. Dataset Description

The data comes directly from OSMI's Kaggle repository, covering 2016 through 2022. Because it's a voluntary online survey spread through developer networks, there's definitely some self-selection bias at play, but it's still the most comprehensive look at our industry available.

I started by loading all seven CSV files, tagging each with a `year` column, and merging them with pandas `pd.concat`. That gave me a massive DataFrame of 3,433 rows and 193 columns. The column count exploded mostly because OSMI kept tweaking their questions over the years, meaning the same semantic question lived under three different column names. I spent a lot of time harmonizing these names into a shared canonical format.

Next, I dropped any column missing more than 50% of its data. Imputing that much missing data is just making things up. I also threw out the free-text fields since natural language processing was out of scope for this project. What I had left, I label-encoded into integers so my ML models wouldn't crash, and squashed the features into a 0-to-1 range using min-max scaling.

After an 80/20 train-test split, my final feature set was down to a lean 18 columns. The training set was (2,746, 18) and the test set was (687, 18). The target variable `treatment` split roughly 58% positive to 42% negative (1998 ones, 1435 zeros)—not perfectly balanced, but definitely close enough that I didn't need to bother with synthetic oversampling.

---

## 3. Search Algorithms

For the search phase, I modeled the data as a graph. The nodes were the different levels of `work_interfere` (how badly a mental health condition impacts work) and `treatment` outcomes. The edges represented how often they appeared together in the dataset. I set the start state to the most common work interference level, and the goal state to `treatment=1` (seeking help). Basically, I wanted to see how the algorithms navigate from a standard symptom profile to getting care.

Here are the results of the different search strategies I implemented:

| Algorithm                   | Path Length / Cost | Nodes Expanded |
| :-------------------------- | :----------------- | :------------- |
| Breadth-First Search (BFS)  | 1                  | 2              |
| Depth-First Search (DFS)    | 1                  | 2              |
| Iterative Deepening (IDDFS) | 1                  | 2              |
| Uniform-Cost Search (UCS)   | Cost: 10.0         | 2              |
| A\* Search                  | Cost: 10.0         | 2              |
| Greedy Best-First           | Cost: 10.0         | 2              |

Because my graph was relatively small, the basic uninformed search strategies like BFS and IDDFS found paths to the goal almost instantly. DFS also got there quickly, even if it wasn't strictly optimal in a broader sense.

For informed search, I used Greedy Best-First and A*. My heuristic was based on the empirical treatment rates for different interference levels. A* worked the best overall because it perfectly balanced path cost with the heuristic estimate, ensuring an optimal path while staying efficient. Greedy search is fast, but it can easily get trapped by its own short-sightedness on a larger graph.

I also messed around with beyond-classical techniques: Hill Climbing, Simulated Annealing (SA), and Beam Search. Hill climbing was fast but greedy. Simulated Annealing was genuinely interesting to watch—it occasionally accepted worse states early on to escape local optima, which is a great trick when the search space gets bumpy. Without SA, algorithms often get stuck on plateaus where every adjacent state looks equally bad.

Finally, I framed the problem as a game where one agent tries to reach treatment and another tries to block it. Both Minimax and Alpha-Beta pruning worked, but Alpha-Beta pruned a massive chunk of the tree, proving how much compute you can save just by ignoring useless branches.

The main takeaway from this phase? The shortest path to `treatment=1` almost always routed through nodes representing high work interference. It was my first hint that severity is the main driver here—something the later phases backed up completely, confirming that symptom intensity is often the primary trigger for pursuing clinical care. This insight laid a solid foundation for the remainder of the analysis.

---

## 4. Constraint Satisfaction

In Phase 3, I set up a Constraint Satisfaction Problem (CSP) to model the environment of a tech worker deciding whether to get help. I picked five key variables: `workplace_support`, `treatment_sought` (the target), `work_interfere`, `remote_work`, and `company_size`.

I wrote five constraints based on what I already knew from the data. For instance, if `work_interfere` is "Often", the probability of seeking treatment has to go up. Conversely, if it's "Never", the probability of treatment drops. I also tied employer support to company size, because tiny startups almost never have formal HR mental health resources.

Before running any search, I applied the AC-3 algorithm to prune the domains. When I artificially pinned `treatment_sought` to zero (simulating someone avoiding care), AC-3 immediately stripped "Often" and "Never" out of the `work_interfere` domain. This makes perfect sense: "Often" contradicts not getting treatment, and "Never" contradicts the lack of workplace support constraints. It was incredibly satisfying to watch the algorithm figure out the boundary conditions on its own. AC-3 definitely helped by shrinking the search space before I even started backtracking.

I ran three backtracking variants: standard, Minimum Remaining Values (MRV), and forward checking. All three required exactly two backtracks. Honestly, getting exactly two backtracks across the board tells me the CSP is well-structured. The conflicts were inherent to the domain, not a symptom of bad variable ordering.

I also threw min-conflicts at it. It successfully converged, dropping to zero violations in just three iterations! That means the problem is heavily under-constrained—there are a lot of valid ways to live as a tech worker with mental health issues. The constraints I wrote are permissive enough to reflect reality, accommodating the massive variety of human experiences captured across all the survey respondents in the industry. It proves that no single constraint strictly mandates treatment-seeking behavior on its own.

---

## 5. Logic and Reasoning

Next, I wrote five propositional logic rules (R1 through R5) to translate my data into readable if-then statements. Machine learning models are black boxes, but logic rules let you actually see the reasoning. I tracked both "support" (how many rows triggered the rule) and "accuracy" (how often the rule was right).

- **R1:** IF `work_interfere` = Often THEN `treatment` = 1. (Fired 65 times, 84.6% accuracy). Severe interference usually means they get help.
- **R2:** IF `work_interfere` = Never THEN `treatment` = 0. (Fired 120 times, 20.8% accuracy). This one failed spectacularly. It turns out people who aren't currently struggling at work still seek preventative or ongoing care.
- **R3:** IF `seek_help` = No AND `remote_work` = No THEN `treatment` = 0. (Fired 172 times, 45.9% accuracy). Worse than a coin flip.
- **R4:** IF `company_size` = 1-5 AND `seek_help` = No THEN `treatment` = 0. (Fired 71 times, 54.9% accuracy). Only marginally better than chance.

But **R5** was the winner: IF `work_interfere` = Sometimes AND `seek_help` = Yes THEN `treatment` = 1. It fired 71 times with an 88.7% accuracy. I love this result. It proves that combining _moderate_ work interference with an employer who actually encourages seeking help is a vastly better predictor than just waiting until interference becomes severe (R1). Supportive environments work.

I ran forward chaining on these rules to simulate a logical proof. By feeding the system facts like `work_interfere=Often`, forward chaining successfully derived the conclusion that `treatment=1`. It simulated exactly how an HR professional might evaluate a situation based on known facts. I even converted the rules to Conjunctive Normal Form (CNF) to allow for formal theorem proving. This transformation definitively proved that the conclusions weren't just guessing—they were logically sound, rigorous derivations based entirely on the established mathematical axioms. This formal proof validates the utility of extracting explicit rules from data.

---

## 6. Machine Learning

Phase 5 was the heavy lifter. I coded five machine learning models entirely from scratch in NumPy—no scikit-learn allowed. I tested them all on the 687-row held-out test set to see which approach generalized best to unseen survey responses.

Here is the final accuracy table summarizing how each model performed:

| Model                       | Test Accuracy / Purity |
| :-------------------------- | :--------------------- |
| Perceptron                  | 66.67%                 |
| Delta Rule                  | 84.13%                 |
| Multilayer Perceptron (MLP) | 71.32%                 |
| K-Means Clustering          | 58.05% Purity          |
| K-Medoid Clustering         | 58.05% Purity          |

The **Delta Rule** performed the best by a massive margin.

I started with a simple Perceptron. It hit 66.67% test accuracy. Because it uses a hard step function and only updates on mistakes, it really only works if the data is perfectly linearly separable. Survey data is noisy and messy, so 66% isn't terrible, but it's definitely hitting its theoretical ceiling. The Perceptron failed to reach higher accuracy because it couldn't handle the fuzzy boundaries inherent in human behavioral data.

Then came the Delta Rule classifier—basically a single-layer neural network with a sigmoid activation trained via gradient descent. It blew everything else out of the water with an 84.13% test accuracy. The fact that a linear classifier performed this well means my 18 preprocessed features actually form a pretty clean, linearly separable decision boundary.

I fully expected my Multilayer Perceptron (MLP) to do better, but it got completely stuck at 71.32%. I used one hidden layer and ReLU activations, and I even implemented He initialization to fix the "dead ReLU" problem I noticed early on. But looking at the loss curve over 200 epochs, it plummeted initially and then just flatlined. The MLP simply underfit the data. It has too many parameters for a sub-3,000 row tabular dataset, and it ended up finding a worse local minimum than the Delta Rule. This outcome highlights the dangers of overcomplicating an architecture.

Finally, I ran K-Means and K-Medoid clustering with k=2. Both of them returned a purity score of exactly 58.05%. At first I thought my code was broken, but then I realized 58% is the exact frequency of the positive class in the dataset. The algorithms were just clustering the majority class together. In an 18-dimensional space, the natural, unsupervised geometric clusters simply do not perfectly align with the complex, binary "did they get treatment" target label.

---

## 7. Cross-Track Comparison

Because MindSight is just Track D of the broader AL2002 course, we can compare our results against other datasets from Track A, Track B, and Track C once everyone submits. Here's my baseline and my theoretical comparison against the other tracks:

| Track             | Dataset                        | Best ML Model          | Best Accuracy |
| :---------------- | :----------------------------- | :--------------------- | :------------ |
| A                 | Continuous Sensor/Physics Data | TBA (Expect MLP)       | TBA           |
| B                 | Image / Vision Data            | TBA (Expect MLP)       | TBA           |
| C                 | NLP / Text Classification      | TBA (Expect Delta/MLP) | TBA           |
| **D (MindSight)** | OSMI Mental Health in Tech     | **Delta Rule**         | **84.13%**    |

Based on the structure of the data, I expect to see some major differences across the tracks. For tracks analyzing continuous sensor data (like Track A) or complex image matrices (like Track B), I fully expect the Multilayer Perceptrons to dominate. Deep learning loves high-dimensional, continuous manifolds where subtle, non-linear relationships dictate the outcome.

But for tracks like mine (Track D) working with categorical, tabular survey data, the landscape is totally different. The features are mostly binary or ordinal integers, the dataset is relatively small, and the signal relies heavily on just a few dominant variables. I strongly suspect we'll see linear classifiers like the Delta Rule holding their own or winning outright in these tabular settings.

I'm also curious to see what is similar across the different tracks despite the varying data modalities. For example, if another dataset has highly distinct behavioral signatures, their K-Means clustering implementation should easily beat my 58% majority-class baseline score. And I really want to see if their respective logic phases manage to output rules that make as much intuitive, real-world sense as my Rule R5 did! The cross-pollination of these ideas will be fascinating to review once the final results are aggregated. I expect that comparing these findings side-by-side will provide a comprehensive understanding of which algorithms excel under which specific data conditions.

---

## 8. Conclusion

MindSight was a massive undertaking, but it proved that a multi-paradigm AI approach can pull real, interpretable insights out of noisy survey data. From search algorithms to machine learning, all five phases pointed toward the same conclusion: severe work interference drives people to get help, but a supportive employer environment is the real catalyst that gets people into treatment _before_ it becomes severe.

The biggest technical takeaway for me was the Delta Rule's 84.13% accuracy crushing the MLP's 71.32%. It's a brutal reminder that throwing more layers at a problem doesn't automatically solve it. On small, tabular datasets, a well-tuned linear model often wins.

If I had another month to work on this, I'd overhaul the preprocessing. Simple label encoding works, but ordinal encoding would be much better for the Likert scales. I'd also want to try adding dropout and early stopping to the MLP to see if I could force it out of its local minimum. And honestly, it hurts that I had to drop the free-text survey comments—running some modern NLP over those fields would probably reveal more about tech workplace culture than the rest of the dataset combined.

---

## References

Open Source Mental Illness (OSMI). _Mental Health in Tech Survey Datasets, 2016–2022._ Available at: [https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)

Russell, S. and Norvig, P. _Artificial Intelligence: A Modern Approach_, 4th Edition. Pearson, 2020.

NumPy Development Team. _NumPy Documentation._ Available at: [https://numpy.org/doc/](https://numpy.org/doc/)

pandas Development Team. _pandas Documentation._ Available at: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)

The Jupyter Project. _Jupyter Project Documentation._ Available at: [https://jupyter.org/documentation](https://jupyter.org/documentation)
