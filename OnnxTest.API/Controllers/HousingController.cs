using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace OnnxTest.API.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class HousingController : ControllerBase
    {
        private readonly InferenceSession _session;

        public HousingController(InferenceSession session)
        {
            _session = session;
        }

        [HttpGet("Score/{mi}/{mha}/{anr}/{anb}/{ao}/{pop}/{lat}/{lng}")]
        public ActionResult<float> Score(float mi, float mha, float anr, float anb,float ao, float pop, float lat, float lng)
        {
            HousingData data = new(mi, mha, anr, anb, ao, pop, lat, lng);
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> result = _session.Run(new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("float_input", data.AsTensor())
            });
            Tensor<float> score = result.First().AsTensor<float>();
            var prediction = new Prediction { PredictedValue = score.First() * 100000 };
            result.Dispose();
            return Ok(prediction);
        }
    }
}
