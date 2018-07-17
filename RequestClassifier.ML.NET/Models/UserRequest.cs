using Microsoft.ML.Runtime.Api;

namespace RequestClassifier.ML.NET
{
    public class UserRequest
    {
        [Column("0")]
        public string Question;
        [Column(ordinal: "1", name: "Label")]
        public bool IsAdministration;
    }
}
