using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.IO;

namespace RequestClassifier.ML.NET
{
    class Program
    {
        static void Main(string[] args)
        {
            var trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "requestClassifier-trainData.tsv");
            var testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "requestClassifier-testData.tsv");
            var modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
            Console.WriteLine("Welcome! Let's predict which department to forward each requests to. As of now we have 2 departments: Administration and Registration");

            Console.WriteLine("Initialize pipeline by loading training data, editing metadata, and selecting ML algorithm");
            var pipeline = new LearningPipeline()
            {
                new TextLoader(trainDataPath).CreateFrom<UserRequest>(useHeader: true),
                new TextFeaturizer("Features", "Question"),
                new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5,  MinDocumentsInLeafs = 2 }
            };

            Console.WriteLine("Let's train our model with all the specs in our learning pipeline and we'll write it to model to disk");
            var model = pipeline.Train<UserRequest, DepartmentPrepiction>();
            model.WriteAsync(modelPath).Wait();

            Console.WriteLine("Let's test our model with test data to see exactly how it performs");
            var testData = new TextLoader(testDataPath).CreateFrom<UserRequest>(useHeader: true);
            var evaluator = new BinaryClassificationEvaluator();
            var metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");

            Console.WriteLine("Now let's try to use our model to with live data");
            do
            {
                Console.WriteLine("Ask a question now");
                string question = Console.ReadLine();

                var prediction = model.Predict(new UserRequest { Question = question });
                model.TryGetScoreLabelNames(out string[] data);
                Console.WriteLine($"Predicted Department: {prediction}");
                Console.WriteLine("Press <ENTER> to continue");
            }
            while (Console.ReadKey().Key == ConsoleKey.Enter);
        }
    }
}
