# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- Configuration for plots ---
plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style

# --- 1. Salary Trend Visualization ---
def plot_salary_trends(job_titles, salaries, trend_type='average', title='Salary Trends'):
    """
    Plots salary trends for different job titles.

    Args:
        job_titles (list or pd.Series): List of job titles.
        salaries (list or pd.Series): Corresponding list of salaries (numeric).
        trend_type (str, optional): 'average' to plot average salary per job title, 
                                  'distribution' to plot boxplots of salary distributions.
                                  Defaults to 'average'.
        title (str, optional): Title of the plot. Defaults to 'Salary Trends'.
    """
    if len(job_titles) != len(salaries):
        print("Error: job_titles and salaries lists must have the same length.")
        return

    df = pd.DataFrame({'job_title': job_titles, 'salary': pd.to_numeric(salaries, errors='coerce')}).dropna()
    if df.empty:
        print("No valid salary data to plot after cleaning.")
        return

    plt.figure(figsize=(12, 7))
    if trend_type == 'average':
        avg_salaries = df.groupby('job_title')['salary'].mean().sort_values(ascending=False)
        avg_salaries.plot(kind='bar')
        plt.ylabel('Average Salary')
        plt.xlabel('Job Title')
    elif trend_type == 'distribution':
        # For better boxplot, sort job titles by median salary or ensure there are enough data points
        order = df.groupby('job_title')['salary'].median().sort_values(ascending=False).index
        sns.boxplot(x='job_title', y='salary', data=df, order=order)
        plt.ylabel('Salary Distribution')
        plt.xlabel('Job Title')
        plt.xticks(rotation=45, ha='right')
    else:
        print(f"Unknown trend_type: {trend_type}. Choose 'average' or 'distribution'.")
        return

    plt.title(title)
    plt.tight_layout()
    plt.show()

# --- 2. Market Segment Visualization (from Clustering) ---
def plot_market_segments(features_for_clustering, cluster_labels, title='Job Market Segments'):
    """
    Visualizes job market segments using dimensionality reduction (e.g., PCA or t-SNE)
    if the features are high-dimensional.

    Args:
        features_for_clustering (np.ndarray): 
            The features used for clustering (e.g., TF-IDF vectors or embeddings).
            If high-dimensional, PCA will be applied for 2D visualization.
        cluster_labels (np.ndarray or list): 
            The cluster labels assigned to each job/document.
        title (str, optional): Title of the plot. Defaults to 'Job Market Segments'.
    """
    if features_for_clustering.shape[0] != len(cluster_labels):
        print("Error: Number of samples in features and cluster_labels must match.")
        return

    num_dimensions = features_for_clustering.shape[1]
    plot_features = features_for_clustering

    if num_dimensions > 2:
        try:
            from sklearn.decomposition import PCA
            print(f"Features have {num_dimensions} dimensions. Applying PCA for 2D visualization...")
            pca = PCA(n_components=2, random_state=42)
            plot_features = pca.fit_transform(features_for_clustering)
            print(f"Explained variance by 2 components: {pca.explained_variance_ratio_.sum():.2f}")
        except ImportError:
            print("PCA requires scikit-learn. Please install it to visualize high-dimensional clusters.")
            print("Plotting first two dimensions as a fallback (if available).")
            if num_dimensions >= 2:
                plot_features = features_for_clustering[:, :2]
            else:
                print("Cannot visualize clusters with less than 2 dimensions without PCA.")
                return
    elif num_dimensions < 2:
        print("Cannot visualize clusters with less than 2 dimensions.")
        return

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.get_cmap('viridis', len(unique_labels))

    for i, label in enumerate(unique_labels):
        plt.scatter(plot_features[cluster_labels == label, 0],
                    plot_features[cluster_labels == label, 1],
                    color=colors(i), label=f'Cluster {label}', alpha=0.7)

    plt.title(title)
    plt.xlabel('Component 1' if num_dimensions > 2 else 'Feature 1')
    plt.ylabel('Component 2' if num_dimensions > 2 else 'Feature 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- 3. Match Quality Visualization (e.g., distribution of match scores) ---
def plot_match_quality_distribution(match_scores, title='Distribution of Resume-Job Match Scores'):
    """
    Plots the distribution of match scores.

    Args:
        match_scores (list or np.ndarray): A list or array of match scores.
        title (str, optional): Title of the plot. Defaults to 'Distribution of Resume-Job Match Scores'.
    """
    if not isinstance(match_scores, (list, np.ndarray)) or len(match_scores) == 0:
        print("Error: match_scores must be a non-empty list or numpy array.")
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(match_scores, kde=True, bins=20)
    plt.title(title)
    plt.xlabel('Match Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# --- 4. Skill Demand/Frequency Visualization ---
def plot_skill_frequency(skill_counts, top_n=20, title='Top Skills in Demand'):
    """
    Plots a bar chart of skill frequencies.

    Args:
        skill_counts (dict or pd.Series): 
            A dictionary or Pandas Series where keys are skill names and values are their counts/frequencies.
        top_n (int, optional): Number of top skills to display. Defaults to 20.
        title (str, optional): Title of the plot. Defaults to 'Top Skills in Demand'.
    """
    if not isinstance(skill_counts, (dict, pd.Series)) or not skill_counts:
        print("Error: skill_counts must be a non-empty dictionary or Pandas Series.")
        return

    if isinstance(skill_counts, dict):
        skill_counts = pd.Series(skill_counts)
    
    top_skills = skill_counts.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(12, 8))
    top_skills.plot(kind='barh') # Horizontal bar chart for better readability of skill names
    plt.title(title)
    plt.xlabel('Frequency / Count')
    plt.ylabel('Skill')
    plt.gca().invert_yaxis() # Display the highest frequency skill at the top
    plt.tight_layout()
    plt.show()

# --- Sample Usage ---
if __name__ == "__main__":
    print("Starting Visualization Module...")

    # --- Salary Trend Example ---
    print("\n--- Plotting Salary Trends ---")
    sample_job_titles = ['Software Engineer', 'Data Scientist', 'Product Manager', 
                         'Software Engineer', 'Data Scientist', 'Software Engineer']
    sample_salaries = [90000, 120000, 110000, 95000, 125000, 85000]
    plot_salary_trends(sample_job_titles, sample_salaries, trend_type='average', title='Average Salaries by Role')
    plot_salary_trends(sample_job_titles, sample_salaries, trend_type='distribution', title='Salary Distributions by Role')

    # --- Market Segment Example (using dummy features and labels) ---
    print("\n--- Plotting Market Segments (Dummy Data) ---")
    # Simulate some features (e.g., from TF-IDF or embeddings after PCA)
    np.random.seed(42)
    dummy_features = np.random.rand(100, 2) # 100 samples, 2 features for direct plotting
    dummy_labels = np.random.randint(0, 3, 100) # 3 clusters
    plot_market_segments(dummy_features, dummy_labels, title='Job Clusters (2D Dummy Features)')

    # Example with high-dimensional features needing PCA
    dummy_high_dim_features = np.random.rand(150, 10) # 150 samples, 10 features
    dummy_high_dim_labels = np.random.randint(0, 4, 150) # 4 clusters
    try:
        from sklearn.decomposition import PCA # Check if PCA is available for this example
        plot_market_segments(dummy_high_dim_features, dummy_high_dim_labels, title='Job Clusters (PCA Reduced)')
    except ImportError:
        print("Skipping high-dimensional market segment plot as scikit-learn (for PCA) is not installed.")

    # --- Match Quality Distribution Example ---
    print("\n--- Plotting Match Quality Distribution ---")
    sample_match_scores = np.random.beta(a=5, b=2, size=200) # Scores typically skewed towards higher values for good matches
    sample_match_scores = np.clip(sample_match_scores * 100, 0, 100) # Scale to 0-100
    plot_match_quality_distribution(sample_match_scores, title='Distribution of Simulated Match Scores')

    # --- Skill Frequency Example ---
    print("\n--- Plotting Skill Frequency ---")
    sample_skill_counts = {
        'Python': 50, 'Java': 30, 'JavaScript': 45, 'SQL': 40, 'AWS': 35,
        'React': 25, 'Machine Learning': 20, 'Data Analysis': 38, 'Project Management': 15,
        'Docker': 18, 'Kubernetes': 12, 'C++': 10, 'Go': 8, 'Swift': 5
    }
    plot_skill_frequency(sample_skill_counts, top_n=10, title='Top 10 Skills by Frequency')

    print("\nVisualization Module execution finished.")
    print("Note: Plots will be displayed in separate windows. Close them to continue.")