import React from 'react';

const ClassificationResults = ({ data }) => {
  return (
    <div>
      <h3>Classification Results of Test Data: </h3>
      <ul>
        <li>Accuracy: {data.accuracy}</li>
        <li>Precision: {data.precision}</li>
        <li>Recall: {data.recall}</li>
        <li>F1 Score: {data.f1}</li>
        <li>Confusion Matrix: {JSON.stringify(data.confusion_matrix)}</li>
      </ul>
    </div>
  );
};

export default ClassificationResults;
