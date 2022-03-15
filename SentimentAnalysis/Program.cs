using Microsoft.ML;
using System;

namespace SentimentAnalysis
{
    class Program
    {
        static void Main(string[] args)
        {
            //Create ML Context
            var context = new MLContext();

            //Set context for getting data file with sentiment data class
            var data = context.Data.LoadFromTextFile<SentimentData>("stock_data.csv", hasHeader: true,
                separatorChar: ',', allowQuoting: true);

            //Assign boolean true or false to pipeline to identify senitment status
            //Use tansforms expression to tranform negative (-1) to false and postive (1) to true
            var pipeline = context.Transforms.Expression("Label", "(x) => x == 1 ? true : false", "Sentiment")
                //transform senitment data to featurised text 
                .Append(context.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text)))
                .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression());

            //create pipline model 
            var model = pipeline.Fit(data);

            //this engine is for assigning an input and output schema for the data prediction results
            var predictionEngine = context.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

           // prediction for a postivie sentiment
             var prediction = predictionEngine.Predict(new SentimentData { Text = "Investing in Tech Companies is good." });

            //display the sentiment status (true) with the probability 
            Console.WriteLine($"First Text Sentiment - {prediction.Prediction} with probability of - {prediction.Probability}");

            //prediction for a negative sentiment 
            var newPrediction = predictionEngine.Predict(new SentimentData { Text = "Twitter is losing a lot stock." });

            //display the sentiment status (false) with the probability 
            Console.WriteLine($"Second Text Sentiment - {newPrediction.Prediction} with probability of - {newPrediction.Probability}");

            //prediction to find netural sentiment
            var anotherPrediction = predictionEngine.Predict(new SentimentData { Text = "Students are not sure with uni." });

            //switch sentiment prediction according to the percentage figures assigned
            switch (anotherPrediction.Probability)
            {
                //assign floats to make probability more granualar in results
                case float sp when sp < .5:
                    Console.WriteLine($"Students sentiment is Negative with probability of {sp}");
                    break;
                case float sp when sp >= .5 && sp <= .7:
                    Console.WriteLine($"Students sentiment is Neutral with probability of {sp}");
                    break;
                case float sp when sp > .7:
                    Console.WriteLine($"Students sentiment is Positive with probability of {sp}");
                    break;
                default:
                    break;
            }
        }
    }
}