// FileUpload.js
import React from 'react';

function FileUpload({ label, onChange }) {
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    onChange(file);
  };

  return (
    <div>
      <label>{label}</label>
      <input type="file" accept=".csv" onChange={handleFileChange} />
    </div>
  );
}

export default FileUpload;