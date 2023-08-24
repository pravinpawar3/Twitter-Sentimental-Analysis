import React, { useState, useEffect } from 'react';
import FileUpload from './Components/FileUpload';
import DashboardAnalysis from './Components/DashboardAnalysis';
import ClassificationResults from './Components/ClassificationResults'; // Adjust the import path as needed
import './App.css';

function App() {
  const [trainFile, setTrainFile] = useState(null);
  const [testFile, setTestFile] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [isSaveData, setIsSaveData] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isExtracting, setIsExtracting] = useState(false);
  const [isDeleteData, setIsDeleteData] = useState(false);
  const [options, setOptions] = useState([]);
  const [selectedOption, setSelectedOption] = useState('');
  const [comment, setComment] = useState('No model found!');
  const [activeSection, setActiveSection] = useState('upload');
  const [modelDir, setModelDir] = useState('upload');
  const [text, setText] = useState('');
  const [predictedLabel, setPredictedLabel] = useState('');
  const [testPerformed, setTestPerformed] = useState(false);
  const [classificationData, setClassificationData] = useState(null);


  const handleFolderSelect = (event) => {
    setModelDir(event.target.value);
  };

  // Replace this with your actual label mapping
  function mapIndexToLabel(predictedClassIndex) {
    const labelMappings = {
      2: 'Negative',
      0: 'Neutral',
      1: 'Positive'
      // Add more mappings as needed
    };

    return labelMappings[predictedClassIndex] || 'Unknown';
  };


  const handlePredict = async () => {
    try {
      const predictText = {
        "data": text
      };
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(predictText) // Send the input text for prediction
      });

      if (response.ok) {
        const data = await response.json();
        const predictedClass = data.predicted_class;
        // Assuming you have a function to map class index to label
        const label = mapIndexToLabel(predictedClass); // Replace with your mapping function

        setPredictedLabel(label);
      } else {
        console.error('Error predicting label');
      }
    } catch (error) {
      console.error('An error occurred:', error);
    }
  };

  // Fetch options from the FastAPI endpoint
  useEffect(() => {
    async function fetchOptions() {
      try {
        const response = await fetch('http://localhost:8000/files/data'); // Replace with your actual API endpoint
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        const data = await response.json();
        setOptions(data.filenames);
      } catch (error) {
        console.error('Error fetching options:', error);
      }
    }

    fetchOptions();
  }, []);

  const handleSelect = (event) => {
    setSelectedOption(event.target.value);
  };


  const handleTrainModel = async () => {
    try {
      setIsSaveData(true);
      setIsTraining(true);
      // Prepare the form data
      const response = await fetch('http://localhost:8000/train/' + selectedOption);
      if (response.ok) {
        setComment('Model training completed!');
        setIsSaveData(false);
        console.log('Data saved successfully');
      } else {
        // Handle error
        console.error('Error saving data');
        setIsSaveData(false);
      }
      setIsTraining(false);
    } catch (error) {
      console.error('An error occurred:', error);
      setIsSaveData(false);
      setIsTraining(false);
    }
  };

  const handleTestModel = async () => {
    try {
      setIsSaveData(true);
      setIsLoading(true);
      // Prepare the form data
      const response = await fetch('http://localhost:8000/test/' + selectedOption);
      if (response.ok) {
        const responseData = await response.json();
        setComment('Model loading completed!');
        setIsSaveData(false);
        // Simulate performing the test and obtaining classificationData
        const fakeClassificationData = {
          accuracy: responseData.accuracy,
          precision: responseData.precision,
          recall: responseData.recall,
          f1: responseData.f1,
          confusion_matrix: responseData.confusion_matrix
        };
        setClassificationData(fakeClassificationData);
        setTestPerformed(true);
        console.log('Data saved successfully');
      } else {
        // Handle error
        console.error('Error saving data');
        setIsSaveData(false);
      }
      setIsLoading(false);
    } catch (error) {
      console.error('An error occurred:', error);
      setIsSaveData(false);
      setIsLoading(false);
    }
  };

  const handleLoadModel = async () => {
    try {
      setIsSaveData(true);
      setIsLoading(true);
      // Prepare the form data
      const payload = {
        "data": modelDir
      };
      const response = await fetch('http://localhost:8000/load', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (response.ok) {
        setComment('Model loading completed!');
        setIsSaveData(false);
        console.log('Data saved successfully');
      } else {
        // Handle error
        console.error('Error saving data');
        setIsSaveData(false);
      }
      setIsLoading(false);
    } catch (error) {
      console.error('An error occurred:', error);
      setIsSaveData(false);
      setIsLoading(false);
    }
  };
  const handleExtractModel = async () => {
    try {
      setIsSaveData(true);
      setIsExtracting(true);
      // Prepare the form data
      const payload = {
        "data": modelDir
      };

      const response = await fetch('http://localhost:8000/extractModel', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (response.ok) {
        setComment('Model extraction completed!');
        setIsSaveData(false);
        console.log('Data saved successfully');
      } else {
        // Handle error
        console.error('Error saving data');
        setIsSaveData(false);
      }
      setIsExtracting(false);
    } catch (error) {
      console.error('An error occurred:', error);
      setIsSaveData(false);
      setIsExtracting(false);
    }
  };

  const handleSaveData = async () => {
    try {
      setIsSaveData(true);

      // Prepare the form data
      const formData = new FormData();
      formData.append('data_type', 'csv'); // Replace with your data type identifier
      formData.append('file', trainFile); // Assuming trainFile is the selected file
      const response = await fetch('http://localhost:8000/upload/data', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const newOption = trainFile.name; // Assuming trainFile is a File object
        setOptions(options => [...options, newOption]);
        await new Promise(resolve => setTimeout(resolve, 1000));
        setIsSaveData(false);
        console.log('Data saved successfully');
      } else {
        // Handle error
        console.error('Error saving data');
        setIsSaveData(false);
      }
    } catch (error) {
      console.error('An error occurred:', error);
      setIsSaveData(false);
    }
  };

  const handleDeleteData = async () => {
    try {
      setIsDeleteData(true);

      // Prepare the form data
      const formData = new FormData();
      formData.append('data_type', 'csv'); // Replace with your data type identifier
      formData.append('file', trainFile); // Assuming trainFile is the selected file
      const response = await fetch('http://localhost:8000/delete/data/' + selectedOption, {
        method: 'DELETE',
      });

      if (response.ok) {
        // Remove the deleted file from options
        setOptions(options => options.filter(option => option !== selectedOption));

        // Reset selectedOption if it matches the deleted filename
        setSelectedOption('');
        await new Promise(resolve => setTimeout(resolve, 1000));
        setIsDeleteData(false);
        console.log('Data deleted successfully');
      } else {
        // Handle error
        console.error('Error deleting data');
        setIsDeleteData(false);
      }
    } catch (error) {
      console.error('An error occurred:', error);
      setIsDeleteData(false);
    }
  };

  return (
    <div className="app">
      <div className="header">
        <h1>Sentiment Analyzer Dashboard</h1>
      </div>
      <div className="sidebar">
        <ul>
          <li
            className={activeSection === 'upload' ? 'active' : ''}
            onClick={() => setActiveSection('upload')}
          >
            Data Management
          </li>
          <li
            className={activeSection === 'train-test' ? 'active' : ''}
            onClick={() => setActiveSection('train-test')}
          >
            Train & Test Model
          </li>
          <li
            className={activeSection === 'analysis' ? 'active' : ''}
            onClick={() => setActiveSection('analysis')}
          >
            Analysis
          </li>
        </ul>
      </div>
      <main className="app-content">
        {activeSection === 'upload' && (
          <>
            <h1 className="app-title">Sentiment Analyzer</h1>
            <FileUpload label="Upload Data: " onChange={setTrainFile} />
            <div className="button-container">
              <button
                className={`save-button`}
                onClick={handleSaveData}
                disabled={isSaveData}
              >
                Save Data
              </button>
              {isSaveData && (
                <div className="training-label">Saving Data...⌛</div>
              )}
            </div>
            <br />
            <br />
            <div className="dropdown-container">
              <label> Delete Data:</label> <></>
              <select value={selectedOption} onChange={handleSelect}>
                <option value="">Select an option</option>
                {options.map((option, index) => (
                  <option key={index} value={option}>
                    {option}
                  </option>
                ))}
              </select>
              {selectedOption && <p className="selected-option">Selected: {selectedOption}</p>}
            </div>

            <br />
            <div className="button-container">
              <button
                className={`Delete Data`}
                onClick={handleDeleteData}
                disabled={isDeleteData}
              >
                Delete
              </button>
              {isDeleteData && (
                <div className="training-label">Deleting Data...⌛</div>
              )}
            </div>

          </>
        )}
        {activeSection === 'train-test' && (
          <>
            <h1 className="app-title">Train & Test </h1>
            <div className="button-container">
              <div className="dropdown-container">
                <label> Select Data:</label> <></>
                <select value={selectedOption} onChange={handleSelect}>
                  <option value="">Select an option</option>
                  {options.map((option, index) => (
                    <option key={index} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </div>
              <br />
              <button
                className={`train-button ${isTraining ? 'loading' : ''}`}
                onClick={handleTrainModel}
                disabled={isTraining || isTesting}
              >
                Train Model
              </button> <></>
              <button
                className={`test-button ${isTesting ? 'loading' : ''}`}
                onClick={handleTestModel}
                disabled={isTesting || isTraining}
              >
                Test Model
              </button>
            </div>
            <br />
            <div>
              {testPerformed && classificationData && (
                <ClassificationResults data={classificationData} />
              )}
            </div>
            <br />
            <label>Predict : </label>
            <br />
            <input
              placeholder="Enter text..."
              value={text}
              onChange={(e) => setText(e.target.value)}
            /><></>
            <button onClick={handlePredict}>Predict</button>
            {predictedLabel && <div>Predicted Label: {predictedLabel}</div>}
            <br />
            <br />
            <br />
            <br />
            <div className="button-container">
              <label>Model Directory: </label>
              <input
                type="text"
                value={modelDir}
                onChange={handleFolderSelect}
                placeholder="Model Directory Path "
              />
              <br /><br />
              <button
                className={`train-button ${isTraining ? 'loading' : ''}`}
                onClick={handleLoadModel}
                disabled={isTraining || isTesting}
              >
                Load Model
              </button>
              <></> <></> <></>
              <button
                className={`test-button ${isTesting ? 'loading' : ''}`}
                onClick={handleExtractModel}
                disabled={isTesting || isTraining}
              >
                Extract Model
              </button>
            </div>
            <br />
            <label>Status:</label>
            {!isTraining && !isTesting && <div className="label">{comment}</div>}
            {isTraining && <div className="training-label">Training...⌛</div>}
            {isTesting && <div className="testing-label">Testing...⌛</div>}
            {isLoading && <div className="training-label">Loading...⌛</div>}
            {isExtracting && <div className="testing-label">Extracting...⌛</div>}
          </>
        )}
        {activeSection === 'analysis' && (
          <>
            <DashboardAnalysis trainFile={trainFile} testFile={testFile} />
          </>
        )}
      </main>
    </div>
  );
}

export default App;
