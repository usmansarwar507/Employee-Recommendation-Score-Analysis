#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_csv("C:/Users/shining star/Downloads/datafile.csv")
data.head()


# In[2]:


# Data processing and cleaning

# Check for missing values in the dataset
missing_values = data.isnull().sum()

# Converting Likert scale responses from strings to numerical values
# We'll define a mapping from the Likert scale responses to numerical values
likert_mapping = {
    'Strongly disagree': 1,
    'Disagree': 2,
    'Neither agree nor disagree': 3,
    'Agree': 4,
    'Strongly agree': 5
}

# Apply the mapping to the Likert scale columns
likert_columns = data.columns[3:15]  # The Likert scale responses are from the 4th to the 12th column
for col in likert_columns:
    data[col] = data[col].map(likert_mapping)

# Check the unique values in the 'What is your department?' column to ensure proper categorization
unique_departments = data['What is your department?'].unique()

# Summary of the data to understand the distribution of responses
summary = data.describe()

missing_values, unique_departments, summary


# In[3]:


data['This last year, I have had opportunities at work to learn and grow.'].unique()


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns

# For cleaner plots, we can set a theme for seaborn
sns.set_theme(style="whitegrid")

# Dropping rows with missing values for a cleaner data exploration
# This is a straightforward approach, but depending on the analysis, imputation might be a better method
cleaned_data = data.dropna()

# We'll create plots for each of the Likert scale questions to see their distribution
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(25, 20))

# Flattening the axes array for easy iteration
axes = axes.flatten()

for i, col in enumerate(likert_columns):
    sns.countplot(x=col, data=cleaned_data, ax=axes[i], palette='viridis')
    axes[i].set_title(col, fontsize=14)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Count')
    axes[i].tick_params(labelsize=12)

# Adjust the layout
plt.tight_layout()
plt.show()





# Employees generally agree or strongly agree with positive statements about their work environment, indicating favorable conditions or sentiments in aspects such as having the right materials, receiving recognition, and feeling their job is important. Areas with lower agreement might indicate potential for improvement.

# In[5]:


# Now let's explore the recommendation score
plt.figure(figsize=(10, 5))
sns.countplot(x='How likely is it that you would recommend us to a friend or colleague?', data=cleaned_data, palette='coolwarm')
plt.title('Distribution of Recommendation Scores', fontsize=16)
plt.xlabel('Recommendation Score', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# A large number of employees have given high recommendation scores (9 or 10), suggesting that they are likely to recommend the company to friends or colleagues. Fewer employees have given lower scores, indicating less satisfaction or willingness to recommend the company.

# In[6]:


# To see if there's an apparent difference between the two employee groups (grade_f and grade_s)
# We'll visualize the distribution of recommendation scores for both groups
plt.figure(figsize=(10, 5))
sns.boxplot(x='Which best describes your current position?', y='How likely is it that you would recommend us to a friend or colleague?', data=cleaned_data, palette='Set2')
plt.title('Recommendation Score by Employee Group', fontsize=16)
plt.xlabel('Employee Group', fontsize=14)
plt.ylabel('Recommendation Score', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# The box plot suggests that there is a variation in how likely employees from different groups are to recommend the company, with some potential outliers, particularly in the grade_s group. Both groups have a median score above the mid-point of the scale, which is positive.

# In[7]:


# Visualize the distribution of recommendation scores by department
plt.figure(figsize=(15, 7))
sns.boxplot(x='What is your department?', y='How likely is it that you would recommend us to a friend or colleague?', data=cleaned_data, palette='Set3')
plt.title('Recommendation Score by Department', fontsize=16)
plt.xlabel('Department', fontsize=14)
plt.ylabel('Recommendation Score', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# The box plot across departments shows variability in the likelihood of employees recommending the company. Some departments have a wider spread of scores, indicating more varied perceptions within those departments.

# In[8]:


# Separate bar chart for recommendation score by department
plt.figure(figsize=(18, 8))
department_score = cleaned_data.groupby('What is your department?')['How likely is it that you would recommend us to a friend or colleague?'].mean().sort_values()
department_score.plot(kind='bar', color='teal')
plt.title('Average Recommendation Score by Department', fontsize=16)
plt.xlabel('Department', fontsize=14)
plt.ylabel('Average Recommendation Score', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Separate bar chart for recommendation score by employee group
plt.figure(figsize=(10, 6))
group_score = cleaned_data.groupby('Which best describes your current position?')['How likely is it that you would recommend us to a friend or colleague?'].mean().sort_values()
group_score.plot(kind='bar', color='coral')
plt.title('Average Recommendation Score by Employee Group', fontsize=16)
plt.xlabel('Employee Group', fontsize=14)
plt.ylabel('Average Recommendation Score', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
plt.show()



# The bar chart indicates that while there's some variation in average recommendation scores between departments, all scores are relatively high. This suggests a generally positive disposition across the company, with no department significantly lagging behind.

# The bar chart shows both employee groups have similar average scores, suggesting that the likelihood of recommending the company is relatively uniform across these groups.

# In[9]:


from scipy.stats import ttest_ind
#define groups
group1 = data[data['Which best describes your current position?']==' grade_f']
group2 = data[data['Which best describes your current position?']=='grade_s']

#test
print(ttest_ind(group1.dropna()['How likely is it that you would recommend us to a friend or colleague?'], group2.dropna()['How likely is it that you would recommend us to a friend or colleague?'], equal_var=True))


# The t-test findings (p-value > 0.05) suggest that any difference in recommendation scores between the two employee groups happened by chance, and isn't a strong enough pattern to conclude that the groups are truly different in this respect. The scores from both groups are statistically similar. Thus, the employee group (whether someone is in grade_f or grade_s) does not seem to be a decisive factor in predicting how likely they are to recommend the company.

# In[10]:


# To create the correlation matrix, we first need to ensure that all relevant columns are in numeric format.
# The recommendation score is already numeric, so we only need to consider the Likert scale variables.

# Calculate the correlation matrix including the recommendation score
correlation_matrix = cleaned_data[likert_columns.tolist() + ['How likely is it that you would recommend us to a friend or colleague?']].corr()

# Plot the correlation matrix using seaborn heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='Blues', cbar_kws={'label': 'Correlation coefficient'})
plt.title('Correlation Matrix between Likert Scale Variables and Recommendation Score', fontsize=16)
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# ### Random Forest Model

# In[12]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Load the dataset

data = pd.read_csv("C:/Users/shining star/Downloads/datafile.csv")

# Define the Likert scale questions and their mapping
likert_columns = data.columns[3:15]  # The Likert scale responses are from the 4th to the 15th column
likert_mapping = {
    'Strongly disagree': 1,
    'Disagree': 2,
    'Neither agree nor disagree': 3,
    'Agree': 4,
    'Strongly agree': 5
}

# Convert Likert scale responses to numeric values
likert_columns_numeric = data[likert_columns].applymap(lambda x: likert_mapping.get(x, x))

# One-hot encode the categorical variables ('department' and 'employee group')
encoder = OneHotEncoder(sparse=False)
encoded_categorical = encoder.fit_transform(data[['What is your department?', 'Which best describes your current position?']])

# Generating feature names for the encoded columns
categories = encoder.categories_
encoded_feature_names = []
for cat, labels in zip(['department', 'employee group'], categories):
    encoded_feature_names.extend([f"{cat}_{label}" for label in labels])

# Convert the encoded data to a DataFrame
encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoded_feature_names)

# Combine the encoded categorical data with the numeric Likert scale data
X_combined = pd.concat([likert_columns_numeric, encoded_categorical_df], axis=1)

# Target variable: Recommendation score
y_combined = data['How likely is it that you would recommend us to a friend or colleague?']

# Drop rows with any missing values in the combined dataset
combined_dataset = pd.concat([X_combined, y_combined], axis=1)
combined_dataset_clean = combined_dataset.dropna()

# Split the cleaned dataset into features and target variable
X_final = combined_dataset_clean.iloc[:, :-1]  # all columns except the last one
y_final = combined_dataset_clean.iloc[:, -1]   # only the last column

# Splitting the final dataset into training and testing sets
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor model
rf_model_final = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_final.fit(X_train_final, y_train_final)

# Predict on the test set
y_pred_final = rf_model_final.predict(X_test_final)

# Evaluate the model
mse_rf_final = mean_squared_error(y_test_final, y_pred_final)
r2_rf_final = r2_score(y_test_final, y_pred_final)

# Extracting feature importance
feature_importance_rf_final = pd.Series(rf_model_final.feature_importances_, index=X_train_final.columns).sort_values(ascending=False)

# Outputs
mse_rf_final, r2_rf_final, feature_importance_rf_final.head(10)


# ### Gradient Boosting Regressor

# In[13]:


from sklearn.ensemble import GradientBoostingRegressor

# Initialize the Gradient Boosting Regressor model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Train the model using the same training set as for the Random Forest
gb_model.fit(X_train_final, y_train_final)

# Predict on the test set
y_pred_gb = gb_model.predict(X_test_final)

# Evaluate the model
mse_gb = mean_squared_error(y_test_final, y_pred_gb)
r2_gb = r2_score(y_test_final, y_pred_gb)

# Extracting feature importance
feature_importance_gb = pd.Series(gb_model.feature_importances_, index=X_train_final.columns).sort_values(ascending=False)

mse_gb, r2_gb, feature_importance_gb.head(10)  # Display top 10 features for brevity


# ## Ridge Regression

# In[18]:


from sklearn.linear_model import Ridge

# Initialize the Ridge Regression model
ridge_model = Ridge(alpha=1.0)  # alpha is the regularization strength

# Train the model using the training set
ridge_model.fit(X_train_final, y_train_final)

# Predict on the test set
y_pred_ridge = ridge_model.predict(X_test_final)

# Evaluate the model performance
mse_ridge = mean_squared_error(y_test_final, y_pred_ridge)
r2_ridge = r2_score(y_test_final, y_pred_ridge)

# Extracting feature importance (coefficients in case of Ridge Regression)
feature_importance_ridge = pd.Series(ridge_model.coef_, index=X_train_final.columns).sort_values(key=abs, ascending=False)

mse_ridge, r2_ridge, feature_importance_ridge.head(10)


# ## Support Vector Machine

# In[19]:


from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# It's often a good idea to scale the features when using Support Vector Machines
svr_model = make_pipeline(StandardScaler(), SVR(kernel='linear'))

# Training the model with the same dataset used in the previous models
svr_model.fit(X_train_final, y_train_final)

# Predicting on the test set
y_pred_svr = svr_model.predict(X_test_final)
# Evaluating the model performance
mse_svr = mean_squared_error(y_test_final, y_pred_svr)
r2_svr = r2_score(y_test_final, y_pred_svr)

# For a linear SVR, the coefficients can give us an idea of feature importance
# However, we need to access the SVR model inside the pipeline
svr_coefs = svr_model.named_steps['svr'].coef_

# In the case of SVR with a linear kernel, the coefficients represent the weight of each feature
feature_importance_svr = pd.Series(svr_coefs.flatten(), index=X_train_final.columns).sort_values(key=abs, ascending=False)

mse_svr, r2_svr, feature_importance_svr.head(10)


# ## Decision Tree

# In[20]:


from sklearn.tree import DecisionTreeRegressor

# Initialize the Decision Tree Regressor model
decision_tree_model = DecisionTreeRegressor(random_state=42)

# Train the model using the same training set as for the previous models
decision_tree_model.fit(X_train_final, y_train_final)

# Predict on the test set
y_pred_decision_tree = decision_tree_model.predict(X_test_final)

# Evaluate the model performance
mse_decision_tree = mean_squared_error(y_test_final, y_pred_decision_tree)
r2_decision_tree = r2_score(y_test_final, y_pred_decision_tree)

# Extracting feature importance
feature_importance_decision_tree = pd.Series(decision_tree_model.feature_importances_, index=X_train_final.columns).sort_values(ascending=False)

mse_decision_tree, r2_decision_tree, feature_importance_decision_tree.head(10)


# 
# 
# ## Random Forest Regressor:
# 
# This model is like asking a bunch of decision trees for their opinion and combining their answers. It's good for more complex patterns.
# After training, it scores about 0.40, similar to Linear Regression, meaning it also explains about 40% of the likelihood to recommend.
# The error score (MSE) here is roughly 1.92, similar to the first model, showing a close average prediction error.
# It also finds that feeling that one's job is important is the top predictor of whether they'd recommend the company, alongside other factors like learning opportunities and knowing what's expected at work.
# 
# ## Gradient Boosting Regressor:
# 
# This model works by building one tree at a time, where each new tree helps correct errors made by the previous ones. It's quite smart and often very effective.
# Its score is a bit lower, around 0.38, meaning it explains 38% of the reasons behind recommendations, which is slightly less than the other models.
# The error score (MSE) is about 1.96, indicating how much the predictions vary from the actual scores.
# Like the other models, it agrees that feeling a sense of purpose in one's job is the strongest predictor for recommending the company.
# In all models, the most influential factor is whether employees feel that their job is important, followed by clarity of expectations and recognition at work. These insights can help a company focus on what matters most to improve employee satisfaction and the likelihood of recommending the workplace.
# 
# ## Ridge Regression
# 
# The Ridge Regression model, with an MSE of 1.85 and an R-squared of 0.42, indicates a moderately good fit to the data. A lower MSE means the model's predictions are relatively close to the actual values, and an R-squared of 0.42 suggests that about 42% of the variation in the recommendation scores is explained by the model. The model identified "The mission or purpose of my organization makes me feel my job is important" as the most influential feature, followed by certain departments like "Grants and Advancement" and "Marketing and Communications." This indicates that both the perception of the job's importance and specific departmental affiliations significantly impact recommendation scores.
# 
# ## Support Vector Machine
# 
# The Support Vector Regression (SVR) model produced an MSE of 1.87 and an R-squared of 0.41, which are similar to the Ridge Regression model, indicating a decent predictive ability. The most influential factors in the SVR model are also led by "The mission or purpose of my organization makes me feel my job is important," highlighting its critical role in determining recommendation scores. Other influential factors include clarity of work expectations and the opportunity to do one's best work.
# 
# ## Decision Tree
# 
# The Decision Tree Regressor shows a higher MSE of 4.00 and a negative R-squared of -0.26. This suggests the model might be overfitting and not generalizing well to new data. The high MSE indicates a greater deviation of the model's predictions from the actual values, and the negative R-squared shows the model's predictions are worse than a simple average. The feature importances highlight "The mission or purpose of my organization makes me feel my job is important" as the most critical predictor, similar to the other models. Other notable features include receiving recognition for good work and opportunities for personal development, suggesting these are key areas influencing employee recommendations.

# ## Summary
# Based on the insights from the machine learning models, here are areas for improvement, potential pain points, and opportunities to increase the likelihood of employees giving recommendation scores of 9-10:
# 
# Mission and Purpose Alignment: Across all models, the belief that the organization's mission or purpose makes the job important was the strongest predictor of recommendation scores. Improving how the company's goals align with employees' values could move recommendation scores higher. Communicating the company's impact, involving employees in goal-setting, and celebrating successes that reflect the company's mission can make employees feel more connected to the purpose of their work.
# 
# Role Clarity and Expectations: Knowing what is expected at work is another key factor. Ensure that job roles are clearly defined, and expectations are communicated effectively. Regular check-ins and clear documentation of roles and responsibilities can help address this.
# 
# Recognition and Appreciation: Recognition for good work in the past week was identified as influential. Implementing or improving employee recognition programs, providing timely and genuine praise, and making sure good work is acknowledged can enhance positive perceptions.
# 
# Development and Growth: Opportunities for learning and growth are important. Fostering a culture of development, providing training resources, and creating clear pathways for career progression can encourage higher recommendation scores.
# 
# Relationships at Work: Having a best friend at work and feeling encouraged by others points to the importance of relationships and support within the company. Building team cohesion and a supportive work environment can make a big difference.
