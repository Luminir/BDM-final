# What we learn in class so far:
## Week 1 — Foundations
1️⃣ Read slides (PDF)
2️⃣ Do practice tasks
3️⃣ explore NumPy & Pandas

## Week 2

### 1. Data Integration
Data integration involves combining data from multiple sources, which presents several challenges:
* **Schema Integration**: Bringing different database schemata together and ensuring equal concepts are represented with equal types
* **Entity Identification**: Matching equal entities across different datasets.
* **Redundancy Analysis**: Identifying repeated entities that need to be removed or filling in rows that are incomplete across sets
* **Value Conflict Resolution**: Handling cases where the same entity has different attribute values in different datasets

### 2. Data Cleaning
Data cleaning is the process of modifying or deleting records until they pass specific **data validity criteria**
* **Validity Constraints**:
    * **Mandatory**: Columns that cannot be empty
    * **Data-type**: Values must match a specific type (e.g., integer, string)
    * **Range**: Values must fall within a set numerical or date range
    * **Unique**: Fields or combinations of fields that must be unique
    * **Cross-field**: Conditions involving multiple fields, such as "discounted price ≤ regular price"
* **Missing Entries**:
    * **MCAR (Missing Completely at Random)**: Independent of any attributes
    * **MAR (Missing at Random)**: Dependent on observable attributes; filling these may introduce bias
    * **MNAR (Missing Not at Random)**: Deterministically dependent on an attribute and is considered informative
    * **Handling Options**: Records can be deleted, or values can be estimated/imputed, though this may introduce errors
* **Inconsistency & Incorrect Entries**: Detected through inconsistency checks (e.g., name vs. abbreviation), domain knowledge (e.g., human age limits), or outlier detection

### 3. Data Transformation & Scaling
* **Normalization**: Necessary because Euclidean distance is meaningless when features have different scales (e.g., age in 10s vs. salary in 1000s).
    * **Standardization (z-scoring)**: Centers data at mean 0 with a standard deviation of 1
    * **Min-Max Scaling**: Constrains data to a [0, 1] range
* **Seasonal Standardization**: A specialized technique used for time-series data, such as ecosystem plant growth, to remove spurious correlations

### 4. Data Reduction
Data reduction aims to decrease space and time complexity, reduce noise, and reveal hidden structures
* **Sampling (Reducing Rows)**:
    * **Uniform Random**: Every item has an equal chance of selection
    * **Stratified**: Data is partitioned into "strata" and sampled within each to ensure rare groups (e.g., millionaires in a population) are represented.
* **Dimensionality Reduction (Reducing Columns)**:
    * **Feature Selection**: Can be supervised (guided by classification performance) or unsupervised (guided by clustering)
    * **Axis Rotation**: Rotating axes (like in PCA or SVD) can remove correlated features and reduce dimensions while minimizing information loss
    '

## Week 3

### 1. Fundamentals of Similarity and Distance
The lecture establishes that "Similarity" is a numerical measure of how alike two data objects are, while "Dissimilarity" (Distance) measures how different they are.

* **Similarity ($s$):** Usually ranges from 0 to 1. Higher values mean higher similarity.
* **Dissimilarity ($d$):** Lower values mean higher similarity.
* **The Proximity Metric:** Proximity refers to either similarity or dissimilarity.

### 2. Distance Metrics for Numeric Data
For continuous or numeric attributes, the primary framework is the **Minkowski Distance**, a generalized distance metric:

$$d(x, y) = \sqrt[n]{\sum_{i=1}^n |x_i - y_i|^p}$$

* **$p = 1$ (Manhattan/City Block Distance):** Sum of absolute differences. Useful for grid-like data.
* **$p = 2$ (Euclidean Distance):** The "as-the-crow-flies" distance. Most common for physical space.
* **$p \to \infty$ (Supremum/Chebyshev Distance):** The maximum difference between any single attribute.



### 3. Similarity Metrics for Categorical & Binary Data
When dealing with "presence or absence" (0/1) data, like market basket analysis or document keywords:

* **Simple Matching Coefficient (SMC):** Used when 0s and 1s are equally important (e.g., Gender).
    * $SMC = \frac{\text{number of matching attributes}}{\text{number of attributes}}$
* **Jaccard Coefficient:** Used for **asymmetric** binary attributes where the "presence" (1) is much more important than the "absence" (0).
    * $J = \frac{M_{11}}{M_{01} + M_{10} + M_{11}}$
    * *Application:* Document similarity, where shared words matter more than the thousands of words neither document contains.

### 4. Cosine Similarity (Non-Binary Vector Data)
This is the standard for high-dimensional data like text mining. It measures the **angle** between two vectors rather than their magnitude.
* It ignores the length of the documents (e.g., a short book and a long book on the same topic will have high cosine similarity).
* Range: 0 (orthogonal/no similarity) to 1 (identical direction).


### 5. Scaling to Big Data: MinHashing & LSH
Calculating similarity between every pair in a massive dataset (e.g., millions of web pages) is $O(n^2)$, which is computationally impossible. The course teaches a two-step optimization:

#### A. MinHashing (Signature Creation)
* **Goal:** Compress large sets into small "signatures" while preserving Jaccard similarity.
* **Logic:** Permute the rows of a boolean matrix. The index of the first row with a '1' for a column is its MinHash value.
* **Property:** The probability that $h(C_1) = h(C_2)$ is exactly equal to the Jaccard similarity $J(C_1, C_2)$.

#### B. Locality-Sensitive Hashing (LSH)
* **Goal:** Find "candidate pairs" without checking all pairs.
* **The "Band" Technique:** 1. Divide the signature matrix into $b$ bands, each with $r$ rows.
    2. Hash each band. If two columns hash to the same bucket in **any** band, they become a candidate pair.
* **The S-Curve:** This creates an "S-curve" probability. You can tune $b$ and $r$ to ensure that pairs with similarity $> s$ are caught, while pairs with similarity $< s$ are filtered out.



### 6. Key Takeaways for your AI Knowledge Base
* **Use Jaccard/MinHash** if your problem involves sets or binary presence (e.g., "Do these two users like the same products?").
* **Use Cosine Similarity** if your problem involves word frequencies or rankings (e.g., "Is this AI-generated text similar to the source material?").
* **Use LSH** if you need to scale the search to thousands of queries per second across millions of records.

## Week 4
### 1. The Market Basket Model
The lecture defines a framework for understanding how items relate to one another within transactions.
* **Items:** The individual entities (e.g., milk, bread, tofu).
* **Transactions:** A set of items purchased together by a single customer or in a single event.
* **Itemset:** A collection of one or more items. A $k$-itemset contains $k$ items.

### 2. Quantifying Association: Key Metrics
To determine if a pattern is "interesting" or "frequent," three primary mathematical filters are used:

* **Support ($supp$):** The fraction of transactions that contain the itemset.
    * *Formula:* $supp(X) = \frac{\text{number of transactions containing } X}{\text{total number of transactions}}$
    * **Minimum Support (minsup):** A threshold set by the user; itemsets meeting this are "frequent."

* **Confidence ($conf$):** Measures how often items in $Y$ appear in transactions that contain $X$. It estimates the probability $P(Y|X)$.
    * *Formula:* $conf(X \Rightarrow Y) = \frac{supp(X \cup Y)}{supp(X)}$
    * **Minimum Confidence (minconf):** A threshold to ensure the rule is reliable.

* **Lift:** Measures how much more likely items in $Y$ are to be purchased given $X$ is purchased, compared to their general popularity.
    * *Formula:* $Lift(X \Rightarrow Y) = \frac{supp(X \cup Y)}{supp(X) \times supp(Y)}$
    * **Interpretation:** Lift > 1 implies a positive correlation; Lift = 1 implies independence; Lift < 1 implies a negative correlation.



### 3. The Apriori Principle
This is the foundational logic for finding frequent itemsets efficiently in big data.
* **The Downward Closure Property:** If an itemset is frequent, all of its subsets must also be frequent.
* **The Pruning Corollary:** If an itemset is **infrequent**, then all of its supersets are also **infrequent**.
    * *Example:* If {Beer} is infrequent, we do not need to calculate the support for {Beer, Diapers} because it is mathematically impossible for it to be frequent.

### 4. The Frequent Itemset Mining Process
The course outlines a multi-pass approach to reduce computational load:
1.  **Pass 1:** Scan all transactions to count support for individual items (1-itemsets). Filter by *minsup*.
2.  **Pass 2:** Generate candidate pairs (2-itemsets) only from the frequent items found in Pass 1. Scan transactions to count their support.
3.  **Subsequent Passes:** Continue joining frequent $(k-1)$-itemsets to create $k$-itemsets, pruning any candidates whose subsets are not frequent.



### 5. Rule Generation
Once frequent itemsets are found, association rules are extracted:
* For every frequent itemset $L$, find all non-empty subsets $f$.
* Output the rule $f \Rightarrow (L - f)$ if the confidence meets the *minconf* threshold.
* Note: A rule like $A, B \Rightarrow C$ is valid only if the itemset $\{A, B, C\}$ is frequent.

### 6. Logic for your AI Knowledge Base
* **Constraint-Based Mining:** When "charging" your AI, emphasize that it must first filter by **support** (to ensure statistical significance) and then by **confidence** (to ensure predictability).
* **Efficiency:** Instruct the AI to use **Apriori pruning** logic. If the AI is looking for patterns in large datasets, it should never evaluate an itemset if a subset has already failed the support threshold.
* **Correlation vs. Co-occurrence:** Use **Lift** to distinguish between items that are truly associated and items that just happen to both be very popular (e.g., bread and milk).

## Week 5
### 1. Two-Phase Mining Process
The lecture divides the mining of association rules into two distinct phases:
* **Phase 1: Frequent Itemset Generation:** Finding all sets of items that satisfy the *minsup* threshold. This is the **computationally expensive** step due to the combinatorial explosion of possible itemsets.
* **Phase 2: Rule Generation:** Extracting rules from the frequent itemsets that satisfy the *minconf* threshold. This step is relatively straightforward and efficient.

### 2. Efficient Frequent Itemset Generation
To handle "Big Data," the course emphasizes reducing the number of candidates ($Ck$) and the number of comparisons.

#### The Apriori Algorithm Refined
The lecture provides a specific implementation strategy:
1.  **Self-Join Step:** Create $C_{k+1}$ by joining $F_k$ with itself. To avoid duplicates and ensure efficiency, only join itemsets that share a **prefix of size $k-1$**.
2.  **Pruning Step:** For every candidate in $C_{k+1}$, check if all its subsets of size $k$ are in $F_k$. If any subset is missing, the candidate is discarded immediately.



### 3. Complexity Bottlenecks
The lecture identifies three main costs in mining:
* **Candidate Generation:** Creating too many potential itemsets (e.g., $10^4$ items can lead to $10^7$ candidate pairs).
* **Support Counting:** Scanning the entire database (transactions $T$) for every candidate.
* **Memory/IO:** Repeated passes over the data stored on disk.

### 4. Advanced Optimization: The Hash Tree
To speed up support counting, the lecture introduces the **Hash Tree** data structure. Instead of comparing every transaction against every candidate, you use a hash tree to quickly identify which candidates are contained in a transaction.
* **Leaf Nodes:** Store the actual candidate itemsets.
* **Interior Nodes:** Contain a hash table that points to children nodes based on the item values.
* **Process:** For a transaction, you generate all possible $k$-itemsets from it and "traverse" the hash tree to increment the support counts of matching candidates.



### 5. Rule Generation Logic
Once $F$ (the union of all frequent itemsets) is found, the logic for generating rules is:
* For each frequent itemset $L$, and each subset $f \subset L$:
* If $\frac{support(L)}{support(f)} \ge minconf$, then output the rule $f \Rightarrow (L - f)$.
* **Efficiency Trick:** If a rule $f \Rightarrow (L - f)$ does not meet *minconf*, then any rule $f' \Rightarrow (L - f')$ where $f' \subset f$ will also fail. This allows for further pruning during rule generation.

### 6. Summary of the Algorithm (as taught by MSc. Bui Quoc Khanh)
The "Complete Logic" for your AI Knowledge Base:
1.  Generate $F_1$ (Singletons).
2.  **While** $F_k$ is not empty:
    * Generate $C_{k+1}$ via **Prefix-Join**.
    * **Prune** $C_{k+1}$ using the Apriori principle.
    * Build a **Hash Tree** for $C_{k+1}$.
    * Scan Transactions and update counts via the tree.
    * Filter $C_{k+1}$ by *minsup* to get $F_{k+1}$.
3.  Generate rules from all $F$ sets using the confidence pruning property.

## Week 6

### 1. The Formal Model
A Recommender System deals with two entities: **Users** and **Items**. 
* **Utility Matrix:** A representation where rows are users, columns are items, and values are "utility" (ratings or preferences).
* **The Goal:** To predict the values in the "holes" (empty cells) of the utility matrix.
* **Cold Start Problem:** New items have no ratings, and new users have no history, making it difficult to provide initial recommendations.

### 2. Content-Based Recommendations
This approach recommends items similar to those a user has liked in the past.
* **Item Profiles:** For each item, create a profile (a vector of features). For movies, features include author, title, actor, or director.
* **User Profiles:** Create a vector describing a user's taste by aggregating the profiles of items they have rated highly.
* **The Algorithm:** Compute the **Cosine Similarity** between the User Profile and potential Item Profiles.
* **Pros/Cons:** Good for niche items; no need for data on other users. However, it suffers from "Overspecialization" (never recommends something outside the user's known profile).

### 3. Collaborative Filtering (CF)
This approach ignores item features and focuses on the relationship between users or between items based on historical data.

#### A. User-User Collaborative Filtering
* Find users $y$ who are similar to user $x$.
* Predict $x$’s rating for item $i$ based on the ratings given to $i$ by the similar users.
* **Similarity Metric:** Usually uses **Pearson Correlation Coefficient** to account for "tough raters" (users who give low scores to everything) and "easy raters."

#### B. Item-Item Collaborative Filtering
* Find items $j$ similar to item $i$ based on who bought/rated them.
* Predict rating for item $i$ based on the user's rating for similar items $j$.
* **Note:** In practice, Item-Item often performs better than User-User because item similarity is more stable than user taste.



### 4. Dimensionality Reduction: Latent Factor Models
When the utility matrix is too sparse, we use **Matrix Factorization** to find hidden (latent) factors that explain ratings.
* **SVD (Singular Value Decomposition):** Decomposes the utility matrix $R$ into $Q \cdot P^T$.
* **Logic:** If we have a matrix of Users $\times$ Movies, latent factors might represent genres like "Sci-Fi," "Romance," or "Action."
* **Optimization:** The goal is to minimize the **RMSE (Root Mean Square Error)** between the known ratings and the predicted ratings ($q_i \cdot p_u$).



### 5. Evaluation Metrics
The course emphasizes that accuracy is not the only metric for a "good" recommender:
* **RMSE:** Standard statistical measure for prediction accuracy.
* **Precision/Recall:** Important for top-K recommendation lists.
* **Diversity:** Recommending a variety of items.
* **Serendipity:** Recommending surprising or unexpected items that the user still likes.

### 6. Logic for your AI Knowledge Base
* **Architecture:** When configuring an AI, have it check for **Content-Based** features first if item metadata is rich, but pivot to **Matrix Factorization (SVD)** for large-scale preference data.
* **Normalization:** Instruct the AI to normalize ratings (subtract the mean) to handle different user rating scales (the "Tough Rater" problem).
* **The S-Curve:** Just like in similarity search, **LSH (Locality-Sensitive Hashing)** can be used here to find "similar users" quickly in a massive utility matrix.

## Week 7
### 1. Definition and Motivation
The lecture defines an **outlier** as an observation that deviates so significantly from other observations that it suggests it was generated by a different mechanism.
* **Context:** While clustering groups similar points together, outlier detection focuses on finding the "different" points.
* **Applications:** Fraud detection (credit cards), medical diagnosis, public health monitoring, and identifying sensor failures.

### 2. Types of Outliers
* **Global Outliers:** A data point that significantly deviates from the rest of the entire dataset.
* **Contextual (Conditional) Outliers:** A data point that is an outlier only in a specific context (e.g., 0°C is normal in winter but an outlier in summer).
* **Collective Outliers:** A subset of data points that deviate significantly from the entire dataset, even if individual points within the subset are not outliers on their own.

### 3. Detection Methodologies

#### A. Extreme Value Analysis (Statistical)
This involves modeling the data distribution and identifying points at the "tails."
* **Z-score:** Measures how many standard deviations a point is from the mean.
* **Box Plots:** Use the Interquartile Range (IQR). Points beyond $1.5 \times IQR$ from the quartiles are potential outliers.


#### B. Distance-Based Outlier Detection
Outliers are defined by their distance from neighbors.
* **DB(ε, π) Outliers:** A point is an outlier if at least fraction $\pi$ of all points are further than distance $\epsilon$ from it.
* **Strengths:** Simple to implement.
* **Weaknesses:** Does not handle datasets with varying densities well.

#### C. Density-Based: Local Outlier Factor (LOF)
This is a more sophisticated approach that identifies outliers relative to their **local neighborhood**.
* **Logic:** A point is an outlier if its local density is significantly lower than the density of its neighbors.
* **Benefit:** It can identify outliers in a dataset that has both high-density and low-density clusters.



#### D. Isolation Forests
Instead of modeling "normal" points, this algorithm explicitly "isolates" anomalies.
* **Mechanism:** It builds multiple random decision trees (Isolation Trees). Because outliers are few and different, they are easier to isolate and thus end up with **shorter path lengths** (closer to the root) than normal points.
* **Anomaly Score:** Calculated based on the average depth $h(x)$.
    * Score close to 1: Likely an outlier.
    * Score below 0.5: Likely a normal point.
* **Extended Isolation Forest:** Allows for random slopes and intercepts during partitioning to provide more freedom in isolating points.


### 4. High-Dimensional Challenges
The lecture notes that in high-dimensional space (Big Data), "distance" becomes less meaningful because almost all pairs of points become nearly equidistant (the **curse of dimensionality**). This makes traditional distance-based methods less effective, necessitating specialized algorithms like Isolation Forests.

### 5. Logic for your AI Knowledge Base
* **Algorithm Selection:** If the data has uniform density, use **Distance-based (ε, π)**. If density varies, use **LOF**. For large-scale, high-dimensional data, use **Isolation Forests**.
* **Isolation Logic:** When implementing an Isolation Forest, instruct the AI to select a random dimension and a random cut-off between the min and max values to partition the data until points are isolated.
* **Scoring:** Use the average path length across several trees to normalize the anomaly score.