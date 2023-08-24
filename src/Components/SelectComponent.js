import React, { useState, useEffect } from 'react';

function SelectComponent({type, onValueChange}) {
  const [options, setOptions] = useState([]);
  const [selectedOption, setSelectedOption] = useState('');

  // Fetch options from the FastAPI endpoint
  useEffect(() => {
    async function fetchOptions() {
      if (type == "Data") {
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
    }
    if (type == "Sentiment") {
      setOptions(['Positive', 'Neutral', 'Negative']);
    }
    fetchOptions();
  }, []);

  const handleSelect = (event) => {
    setSelectedOption(event.target.value);
    onValueChange(event.target.value);
  };

  return (
    <div className="dropdown-container">
      <label>Select Data : </label>
      <select value={selectedOption} onChange={handleSelect}>
        <option value="">Select an option</option>
        {options.map((option, index) => (
          <option key={index} value={option}>
            {option}
          </option>
        ))}
      </select>
    </div>
  );
}

export default SelectComponent;
