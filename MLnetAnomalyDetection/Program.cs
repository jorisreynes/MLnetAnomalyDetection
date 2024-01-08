using Microsoft.ML;
using MLnetAnomalyDetection;

string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "product-sales.csv");
//assign the Number of records in dataset file to constant variable
const int _docsize = 36;


// La classe MLContext est un point de départ pour toutes les opérations ML.NET, et l’initialisation de mlContext crée un environnement ML.NET qui peut être partagé
// par les objets de flux de travail de création de modèle. Sur le plan conceptuel, elle est similaire à DBContext dans Entity Framework
MLContext mlContext = new MLContext();

// IDataView est un moyen flexible et efficace de décrire des données tabulaires (numériques et texte). Les données peuvent être chargées à partir d’un fichier texte
// ou d’autres sources (par exemple, fichiers journaux ou de base de données SQL) dans un objet IDataView.
IDataView dataView = mlContext.Data.LoadFromTextFile<ProductSalesData>(path: _dataPath, hasHeader: true, separatorChar: ',');

DetectSpike(mlContext, _docsize, dataView);


// La méthode CreateEmptyDataView() produit un objet de vue de données vide avec le bon schéma à utiliser comme entrée à la méthode IEstimator.Fit().
IDataView CreateEmptyDataView(MLContext mlContext)
{
    // Create empty DataView. We just need the schema to call Fit() for the time series transforms
    IEnumerable<ProductSalesData> enumerableData = new List<ProductSalesData>();
    return mlContext.Data.LoadFromEnumerable(enumerableData);
}


// Crée la transformation à partir de l’estimateur. Détecte les pics par rapport aux données de ventes historiques. Affiche les résultats.
void DetectSpike(MLContext mlContext, int docSize, IDataView productSales)
{
    // Utilisez IidSpikeEstimator pour entraîner le modèle à la détection de pics
    var iidSpikeEstimator = mlContext.Transforms.DetectIidSpike(outputColumnName: nameof(ProductSalesPrediction.Prediction), inputColumnName: nameof(ProductSalesData.numSales), confidence: 95d, pvalueHistoryLength: docSize / 4);

    // Créez la transformation de détection de pic
    ITransformer iidSpikeTransform = iidSpikeEstimator.Fit(CreateEmptyDataView(mlContext));


    IDataView transformedData = iidSpikeTransform.Transform(productSales);

    // Convertissez votre transformedData en un IEnumerable fortement typé pour faciliter l’affichage
    var predictions = mlContext.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, reuseRowObject: false);

    Console.WriteLine("Alert\tScore\tP-Value\tAverage\tStdDev\tAnomaly");





    Queue<double> lastTenValues = new Queue<double>();
    int queueSize = 10;

    double coef = 2;


    foreach (var p in predictions)
    {


        lastTenValues.Enqueue(p.Prediction[1]);


        if (lastTenValues.Count > queueSize)
        {
            lastTenValues.Dequeue();
        }

        double average = lastTenValues.Average();
        double sumOfSquaresOfDifferences = lastTenValues.Select(val => (val - average) * (val - average)).Sum();
        double standardDeviation = Math.Sqrt(sumOfSquaresOfDifferences / queueSize);
        int isAnomaly = 0;

        if (lastTenValues.Count == queueSize)
        {
            if (p.Prediction[1] < (average - (coef * standardDeviation)) || p.Prediction[1] > (average + (coef * standardDeviation))) { isAnomaly = 1; }

        }

        if (p.Prediction is not null)
        {
            var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}\t{average:f2}\t{standardDeviation:f2}\t{isAnomaly}";

            if (p.Prediction[0] == 1)
            {
                results += " <-- Spike detected";
            }

            Console.WriteLine(results);
        }
    }
    Console.WriteLine("");


}