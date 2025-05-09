using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AICircleDetector.AI
{
    public class AIResult
    {
        public bool Success { get; set; }
        public string Message { get; set; } = string.Empty;
        public float Loss { get; set; } = 0.0f;
        public float Accuracy{ get; set; } = 0.0f;
        public float MAE{ get; set; } = 0.0f;
    }
}
