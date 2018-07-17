using Microsoft.ML.Runtime.Api;

namespace RequestClassifier.ML.NET
{
    public class DepartmentPrepiction
    {
        [ColumnName("PredictedLabel")]
        public bool IsAdministration;

        public override string ToString()
        {
            return IsAdministration ? "Administration" : "Registration";
        }
    }
}
