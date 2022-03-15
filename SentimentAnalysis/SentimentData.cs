using Microsoft.ML.Data;

namespace SentimentAnalysis
{
    public class SentimentData
    {
        //Assign text property for reading text data
        [LoadColumn(0)]
        public string Text { get; set; }

        //Assign sentiment property for reading text data
        [LoadColumn(1)]
        public float Sentiment { get; set; }
    }
}