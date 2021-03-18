using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using System;
using System.Linq;

namespace RoboterRangliste
{
    class Program
    {
        static void Main(string[] args)
        {
            var collisonAvoidedData = new[] {0,1,0,3,4,4,2,3,3};
            var collisonData = new[] {1,2,2,0,0,1,4,4,1};

            var drivingRoute = new Range(collisonAvoidedData.Length);
            var robots = new Range(collisonAvoidedData.Concat(collisonData).Max() + 1);
            var roboterSkills = Variable.Array<double>(robots);
            roboterSkills[robots] = Variable.GaussianFromMeanAndVariance(9, 12).ForEach(robots);

            var collisonAvoided = Variable.Array<int>(drivingRoute);
            var collison = Variable.Array<int>(drivingRoute);

            using (Variable.ForEach(drivingRoute))
            {
                var collisonAvoidedPerformance = Variable.GaussianFromMeanAndVariance(roboterSkills[collisonAvoided[drivingRoute]], 1.0);
                var collisonPerformace = Variable.GaussianFromMeanAndVariance(roboterSkills[collison[drivingRoute]], 1.0);
                Variable.ConstrainTrue(collisonAvoidedPerformance > collisonPerformace);
            }

            collisonAvoided.ObservedValue = collisonAvoidedData;
            collison.ObservedValue = collisonData;

            var inferenceEngine = new InferenceEngine();
            var inferredSkills = inferenceEngine.Infer<Gaussian[]>(roboterSkills);

            var orderedRoboterSkills = inferredSkills
                .Select((s, i) => new { Roboter = i, Skill = s }).OrderByDescending(rs => rs.Skill.GetMean());

            foreach(var roboterSkill in orderedRoboterSkills)
            {
                Console.WriteLine($"Roboter {roboterSkill.Roboter} skill: {roboterSkill.Skill}");
            }

            Console.ReadLine();
        }
    }
}
