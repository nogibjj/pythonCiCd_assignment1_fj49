"""Necessary imports"""
import pandas as pd
import matplotlib.pyplot as plt


# reading the data
def reader(file_name):
    """Reads data"""
    data = pd.read_csv(file_name)
    return data

# basic stats
def mean(column_name:str, data):
    """Calculates the mean"""
    mean_duration = int(data[column_name].mean())
    return mean_duration


def median(column_name, data):
    """Calculates the median"""
    median_duration = int(data[column_name].median())
    return median_duration


def mode(column_name, data):
    """"Calculates the mode"""
    mode_duration = int(data[column_name].mode().iloc[0])
    return mode_duration


def std(column_name, data):
    """Calculates the std"""
    std_duration = int(data[column_name].std())
    return std_duration


# making a plot
def viz(column_name, data, jupyter = False):
    """Creates visualization of the top 10 artists on spotify"""
    value_counts = data[column_name].value_counts()
    top_10_value_counts = value_counts.head(10)
    fig = plt.figure(figsize=(10, 6))
    fig = plt.bar(top_10_value_counts.index, top_10_value_counts.values)
    # Add labels and a title to the plot
    fig = plt.xlabel("Top Artists")
    fig = plt.ylabel("Number of top tracks")
    fig = plt.title("Which artists had the most top tracks in the last few years?")
    #plt.show()
    if not jupyter:
        plt.savefig("output/visualization.png")
    return fig

def summary_report(column_name,data):
    '''generates a report'''
    summary_report_path = r'output/generated_report.md'
    with open(summary_report_path, "w", encoding="utf-8") as report:
        report.write(f'Mean: {mean(column_name, data)} \n \n \n')
        report.write(f'Median: {median(column_name, data)} \n \n \n')
        report.write(f'Standard Deviation: {std(column_name, data)} \n \n \n')

