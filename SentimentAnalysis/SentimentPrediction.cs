using Microsoft.ML.Data;

namespace SentimentAnalysis
{
    public class SentimentPrediction
    {
        //create schema for output prediction results
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}