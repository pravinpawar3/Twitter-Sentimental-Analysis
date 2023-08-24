import React, { useState } from 'react';
import ReactWordcloud from 'react-wordcloud';
import Plot from 'react-plotly.js'; // Import Plotly component
import SelectComponent from './SelectComponent';

function DashboardAnalysis() {
    const [barChartData, setBarChartData] = useState([]);
    const [wordCloudData, setWordCloudData] = useState([]);
    const [pieChartLabel, setPieChartLabels] = useState([]);
    const [pieChartValues, setPieChartValues] = useState([]);
    const [barChartX, setBarChartX] = useState([]);
    const [barChartY, setBarChartY] = useState([]);
    const [type, setType] = useState('');

    const handleValueChangeForBarChart = (selectedValue) => {
       
        let barChartX = null;   
        let barChartY = null;
        if (selectedValue === 'Positive') {
            barChartY= barChartData.positive.values;
            barChartX = barChartData.positive.labels;
        } else if (selectedValue === 'Neutral') {
            barChartY= barChartData.neutral.values;
            barChartX = barChartData.neutral.labels;
        } else if (selectedValue === 'Negative') {
            barChartY= barChartData.negative.values;
            barChartX = barChartData.negative.labels;
        }else{
            return ;
        }
        setBarChartX(barChartX);
        setBarChartY(barChartY);
    };

    const handleValueChangeForWordCloud = (selectedValue) => {
       
        let wordCloudCt = null;
        if (selectedValue === 'Positive') {
            wordCloudCt = wordCloudData.positive;
        } else if (selectedValue === 'Neutral') {
            wordCloudCt = wordCloudData.neutral;
        } else if (selectedValue === 'Negative') {
            wordCloudCt = wordCloudData.negative;
        }else{
            return ;
        }
        setType(selectedValue);
        setWordCloudCt(wordCloudCt);
    };
    const handleAnalyzeClick = async () => {
        try {
            const response = await fetch('http://localhost:8000/fetchData/data/Tweets.csv');
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            const data = await response.json();
            setBarChartData(data.barChartData);
            setWordCloudData(data.wordCloudData);
            setPieChartValues(data.pieChartData.values);
            setPieChartLabels(data.pieChartData.labels);
        } catch (error) {
            console.error('Error fetching data:', error);
        }
    };

    const pieChartData = {
        values: pieChartValues,// sentimentData.map(row => row.count || 0),
        labels: pieChartLabel,
        type: 'pie',
        marker: {
            colors: ['#36A2EB', '#FFCE56', '#FF6384'],
        },
    };


    const [wordCloudCt, setWordCloudCt] = useState([]);

    const barChartCt = {
        x: barChartX,
        y: barChartY,
        type: 'bar',
        marker: { color: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] },
    };

    const handleValueChange = (selectedValue) =>{};
    // Word cloud options
    const wordCloudOptions = {
        colors: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        enableTooltip: true,
        deterministic: false,
        fontFamily: 'impact',
        fontSizes: [15, 60],
        fontStyle: 'normal',
        fontWeight: 'normal',
        padding: 1,
        rotations: 3,
        rotationAngles: [0, 90],
        scale: 'sqrt',
        spiral: 'archimedean',
        transitionDuration: 1000,
    };

    return (
        <div className="data-analyzer">
            <h2>Data Analysis</h2>
            <div className="chart-container">
                <div className="dropdown-container">
                    <SelectComponent type="Data" onValueChange={handleValueChange} lable="Select Data : " />
                </div>
                <br />
                <button onClick={handleAnalyzeClick} style={{ fontSize: '16px' }}>Analyze</button>
                <div className="pie-chart">
                    <h3>Sentiment Distribution</h3>
                    <Plot data={[pieChartData]} />
                </div>
                <div className="bar-chart">
                    <h3>Sentiment per Target Class</h3>
                    <SelectComponent type="Sentiment" onValueChange={handleValueChangeForBarChart} lable="Data : " />
                    <Plot data={[barChartCt]} />
                </div>
                <div className="word-cloud">
                    <h3>Word Cloud 
                    <SelectComponent type="Sentiment" onValueChange={handleValueChangeForWordCloud} lable="Data : " /> </h3>
                    <ReactWordcloud words={wordCloudCt} options={wordCloudOptions} />
                </div>
            </div>
        </div>
    );
}

export default DashboardAnalysis;
