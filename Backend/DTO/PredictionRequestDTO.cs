namespace Mega.DTO
{
    public class PredictionRequestDTO
    {
        public int DatasetId { get; set; }
        public int Version { get; set; }
        public string Model_Name { get; set; } = string.Empty;
        public Dictionary<string, object> Features { get; set; } = new();
    }
}
