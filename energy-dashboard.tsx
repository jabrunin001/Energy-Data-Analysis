import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, ZAxis, Cell, PieChart, Pie, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import _ from 'lodash';

const EnergyDashboard = () => {
  const [data, setData] = useState([]);
  const [regions, setRegions] = useState([]);
  const [regionStats, setRegionStats] = useState({});
  const [correlations, setCorrelations] = useState({});
  const [peakAnalysis, setPeakAnalysis] = useState({});
  const [renewableImpact, setRenewableImpact] = useState({});
  const [selectedRegion, setSelectedRegion] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  // Colors for charts
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];
  const HEALTH_COLORS = ['#c6e48b', '#7bc96f', '#239a3b', '#196127'];
  const CONSUMPTION_COLORS = ['#d4f7ff', '#a2dbfa', '#39a9db', '#1a7db6', '#0d47a1'];

  useEffect(() => {
    // Create synthetic data
    const generateSyntheticData = () => {
      // This would normally be fetched from an API or CSV file
      const data = [];
      const regions = ['Region_1', 'Region_2', 'Region_3', 'Region_4', 'Region_5'];
      
      // Generate two months of hourly data
      for (let day = 0; day < 60; day++) {
        for (let hour = 0; hour < 24; hour++) {
          const timestamp = new Date(2023, 0, 1 + day, hour);
          
          for (const region of regions) {
            // Base consumption pattern with daily cycle
            let baseConsumption = 1000 + 500 * Math.sin((hour - 6) * Math.PI / 12);
            
            // Weekend effect (reduce consumption on weekends)
            const dayOfWeek = timestamp.getDay();
            if (dayOfWeek === 0 || dayOfWeek === 6) {
              baseConsumption *= 0.8;
            }
            
            // Seasonal effect
            const monthEffect = Math.sin((day / 30) * 2 * Math.PI);
            baseConsumption += monthEffect * 200;
            
            // Regional variation
            const regionIndex = regions.indexOf(region);
            const regionMultiplier = 0.8 + (regionIndex * 0.1);
            baseConsumption *= regionMultiplier;
            
            // Random noise
            baseConsumption += 100 * (Math.random() - 0.5);
            
            // Temperature is inversely related to consumption in winter
            const baseTemp = 60 + 20 * Math.sin((day / 365) * 2 * Math.PI);
            const hourlyTempVariation = 10 * Math.sin((hour - 14) * Math.PI / 12);
            const temperature = baseTemp + hourlyTempVariation + 5 * (Math.random() - 0.5);
            
            // Business activity correlates with consumption
            const businessActivity = baseConsumption / 1000 * (0.8 + 0.4 * Math.random());
            
            // Renewable percentage varies by region
            const renewableValue = 20 + regionIndex * 3 + day / 10 + 10 * (Math.random() - 0.5);
            const renewablePercentage = Math.min(Math.max(renewableValue, 5), 75);
            
            // Hospital admissions have correlation with extreme temperatures
            let admissionBase = 100;
            if (temperature > 85 || temperature < 30) {
              admissionBase += Math.abs(temperature - 75) / 5;
            }
            const hospitalAdmissions = Math.round(admissionBase * (0.8 + 0.4 * Math.random()));
            
            data.push({
              region,
              timestamp,
              consumption: baseConsumption,
              temperature,
              businessActivity,
              renewablePercentage,
              hospitalAdmissions
            });
          }
        }
      }
      
      return { data, regions };
    };

    const { data, regions } = generateSyntheticData();
    setData(data);
    setRegions(regions);
    setSelectedRegion(regions[0]);
    
    // Calculate regional statistics
    const stats = {};
    for (const region of regions) {
      const regionData = data.filter(d => d.region === region);
      
      stats[region] = {
        avgConsumption: _.meanBy(regionData, 'consumption'),
        maxConsumption: _.maxBy(regionData, 'consumption').consumption,
        minConsumption: _.minBy(regionData, 'consumption').consumption,
        avgTemperature: _.meanBy(regionData, 'temperature'),
        avgRenewable: _.meanBy(regionData, 'renewablePercentage'),
        avgAdmissions: _.meanBy(regionData, 'hospitalAdmissions')
      };
    }
    setRegionStats(stats);
    
    // Calculate correlations
    const corr = {};
    for (const region of regions) {
      const regionData = data.filter(d => d.region === region);
      
      const calculateCorrelation = (xValues, yValues) => {
        const n = xValues.length;
        const sumX = _.sum(xValues);
        const sumY = _.sum(yValues);
        const sumXY = _.sum(_.zipWith(xValues, yValues, (a, b) => a * b));
        const sumX2 = _.sum(xValues.map(x => x * x));
        const sumY2 = _.sum(yValues.map(y => y * y));
        
        const numerator = n * sumXY - sumX * sumY;
        const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
        
        return numerator / denominator;
      };
      
      const consumption = regionData.map(d => d.consumption);
      const temperature = regionData.map(d => d.temperature);
      const business = regionData.map(d => d.businessActivity);
      const renewable = regionData.map(d => d.renewablePercentage);
      const admissions = regionData.map(d => d.hospitalAdmissions);
      
      corr[region] = {
        consumption_temperature: calculateCorrelation(consumption, temperature),
        consumption_business: calculateCorrelation(consumption, business),
        consumption_renewable: calculateCorrelation(consumption, renewable),
        temperature_admissions: calculateCorrelation(temperature, admissions),
        consumption_admissions: calculateCorrelation(consumption, admissions)
      };
    }
    setCorrelations(corr);
    
    // Analyze peak consumption times and correlations with health outcomes
    const peaks = {};
    for (const region of regions) {
      const regionData = data.filter(d => d.region === region);
      
      // Sort by consumption to find peaks
      const sortedData = [...regionData].sort((a, b) => b.consumption - a.consumption);
      const peakThreshold = sortedData[Math.floor(sortedData.length * 0.05)].consumption;
      
      // Identify peak hours
      const peakHours = regionData.filter(d => d.consumption >= peakThreshold);
      
      // Analyze the relationship between peaks and hospital admissions
      // Get the following day's hospital admissions
      const peakDates = new Set(peakHours.map(d => d.timestamp.toISOString().slice(0, 10)));
      
      // Average hospital admissions on days after peaks vs. regular days
      const admissionsAfterPeaks = [];
      const admissionsRegular = [];
      
      for (let i = 0; i < regionData.length - 24; i++) {
        const currentDate = regionData[i].timestamp.toISOString().slice(0, 10);
        const nextDayAdmissions = _.meanBy(regionData.slice(i + 24, i + 48), 'hospitalAdmissions');
        
        if (peakDates.has(currentDate)) {
          admissionsAfterPeaks.push(nextDayAdmissions);
        } else {
          admissionsRegular.push(nextDayAdmissions);
        }
      }
      
      const avgAdmissionsAfterPeaks = _.mean(admissionsAfterPeaks);
      const avgAdmissionsRegular = _.mean(admissionsRegular);
      
      peaks[region] = {
        peakThreshold,
        numPeakHours: peakHours.length,
        avgAdmissionsAfterPeaks,
        avgAdmissionsRegular,
        percentDifference: ((avgAdmissionsAfterPeaks - avgAdmissionsRegular) / avgAdmissionsRegular) * 100
      };
    }
    setPeakAnalysis(peaks);
    
    // Analyze renewable energy impact on business metrics
    const renewableEffect = {};
    for (const region of regions) {
      const regionData = data.filter(d => d.region === region);
      
      // Group into high renewable and low renewable days
      const sortedByRenewable = [...regionData].sort((a, b) => b.renewablePercentage - a.renewablePercentage);
      const highRenewableThreshold = sortedByRenewable[Math.floor(sortedByRenewable.length * 0.25)].renewablePercentage;
      const lowRenewableThreshold = sortedByRenewable[Math.floor(sortedByRenewable.length * 0.75)].renewablePercentage;
      
      const highRenewableData = regionData.filter(d => d.renewablePercentage >= highRenewableThreshold);
      const lowRenewableData = regionData.filter(d => d.renewablePercentage <= lowRenewableThreshold);
      
      // Compare business activity
      const avgBusinessHighRenewable = _.meanBy(highRenewableData, 'businessActivity');
      const avgBusinessLowRenewable = _.meanBy(lowRenewableData, 'businessActivity');
      
      renewableEffect[region] = {
        highRenewableThreshold,
        lowRenewableThreshold,
        avgBusinessHighRenewable,
        avgBusinessLowRenewable,
        percentDifference: ((avgBusinessHighRenewable - avgBusinessLowRenewable) / avgBusinessLowRenewable) * 100
      };
    }
    setRenewableImpact(renewableEffect);
  }, []);

  // Get daily average consumption by region
  const getDailyConsumptionData = () => {
    const result = [];
    if (data.length === 0) return result;
    
    // Group by date and region
    const grouped = _.groupBy(data, item => {
      const date = new Date(item.timestamp);
      return `${date.toISOString().slice(0, 10)}_${item.region}`;
    });
    
    // Calculate averages
    for (const key in grouped) {
      const [date, region] = key.split('_');
      const avgConsumption = _.meanBy(grouped[key], 'consumption');
      
      result.push({
        date,
        region,
        avgConsumption
      });
    }
    
    // Sort by date
    return _.sortBy(result, 'date');
  };

  // Get hourly pattern by day of week
  const getHourlyPatternData = (region) => {
    if (!region || data.length === 0) return [];
    
    const regionData = data.filter(d => d.region === region);
    const hourlyByDayOfWeek = Array(7).fill().map(() => Array(24).fill(0));
    const countByDayOfWeek = Array(7).fill().map(() => Array(24).fill(0));
    
    for (const record of regionData) {
      const date = new Date(record.timestamp);
      const dayOfWeek = date.getDay();
      const hour = date.getHours();
      
      hourlyByDayOfWeek[dayOfWeek][hour] += record.consumption;
      countByDayOfWeek[dayOfWeek][hour] += 1;
    }
    
    const result = [];
    const dayNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    
    for (let day = 0; day < 7; day++) {
      for (let hour = 0; hour < 24; hour++) {
        if (countByDayOfWeek[day][hour] > 0) {
          result.push({
            day: dayNames[day],
            hour,
            avgConsumption: hourlyByDayOfWeek[day][hour] / countByDayOfWeek[day][hour]
          });
        }
      }
    }
    
    return result;
  };

  // Get correlation data for visualization
  const getCorrelationData = () => {
    if (Object.keys(correlations).length === 0) return [];
    
    const result = [];
    for (const region in correlations) {
      const corr = correlations[region];
      result.push({
        region,
        'Consumption vs Temperature': corr.consumption_temperature,
        'Consumption vs Business': corr.consumption_business,
        'Consumption vs Renewable': corr.consumption_renewable,
        'Temperature vs Health': corr.temperature_admissions,
        'Consumption vs Health': corr.consumption_admissions
      });
    }
    
    return result;
  };

  // Get consumption vs temperature scatter data
  const getConsumptionTemperatureData = (region) => {
    if (!region || data.length === 0) return [];
    
    return data
      .filter(d => d.region === region)
      .map(d => ({
        temperature: d.temperature,
        consumption: d.consumption
      }));
  };

  // Get health impact of peak energy consumption
  const getHealthImpactData = () => {
    if (Object.keys(peakAnalysis).length === 0) return [];
    
    return Object.entries(peakAnalysis).map(([region, data]) => ({
      region,
      regularAdmissions: data.avgAdmissionsRegular,
      peakAdmissions: data.avgAdmissionsAfterPeaks,
      percentIncrease: data.percentDifference
    }));
  };

  // Get business impact of renewable energy
  const getRenewableBusinessImpactData = () => {
    if (Object.keys(renewableImpact).length === 0) return [];
    
    return Object.entries(renewableImpact).map(([region, data]) => ({
      region,
      lowRenewableBusiness: data.avgBusinessLowRenewable,
      highRenewableBusiness: data.avgBusinessHighRenewable,
      percentImprovement: data.percentDifference
    }));
  };

  // Format day and hour for heatmap
  const formatDayHour = (day, hour) => {
    const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    return `${dayNames[day]} ${hour}:00`;
  };

  // Render the dashboard
  // Generate insights based on the data
  const generateInsights = () => {
    if (Object.keys(correlations).length === 0 || 
        Object.keys(peakAnalysis).length === 0 || 
        Object.keys(renewableImpact).length === 0) {
      return [];
    }
    
    const insights = [];
    
    // Find strongest temperature-consumption correlation
    const tempCorr = Object.entries(correlations)
      .map(([region, data]) => ({ region, correlation: data.consumption_temperature }))
      .sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation))[0];
    
    insights.push({
      title: "Temperature-Consumption Relationship",
      text: `${tempCorr.region} shows the strongest temperature-consumption correlation (${tempCorr.correlation.toFixed(3)}), indicating ${tempCorr.correlation < 0 ? "inverse" : "direct"} relationship between temperature and energy consumption.`
    });
    
    // Find region with highest health impact
    const healthImpact = Object.entries(peakAnalysis)
      .map(([region, data]) => ({ region, impact: data.percentDifference }))
      .sort((a, b) => b.impact - a.impact)[0];
    
    insights.push({
      title: "Health Impact Finding",
      text: `${healthImpact.region} shows a ${healthImpact.impact.toFixed(2)}% increase in hospital admissions following peak energy consumption periods, suggesting potential public health considerations.`
    });
    
    // Find region with highest renewable benefit
    const renewableBenefit = Object.entries(renewableImpact)
      .map(([region, data]) => ({ region, benefit: data.percentDifference }))
      .sort((a, b) => b.benefit - a.benefit)[0];
    
    insights.push({
      title: "Renewable Energy Business Impact",
      text: `${renewableBenefit.region} shows a ${renewableBenefit.benefit.toFixed(2)}% improvement in business activity with higher renewable energy adoption, suggesting economic benefits of clean energy.`
    });
    
    return insights;
  };

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <h1 className="text-3xl font-bold mb-6 text-center text-blue-800">Energy Consumption Analytics Dashboard</h1>
      
      {/* Region selector */}
      <div className="mb-6 flex justify-center">
        <div className="bg-white p-4 rounded-lg shadow-md">
          <label className="mr-2 font-semibold">Select Region:</label>
          <select 
            value={selectedRegion || ''} 
            onChange={(e) => setSelectedRegion(e.target.value)}
            className="p-2 border rounded"
          >
            {regions.map(region => (
              <option key={region} value={region}>{region}</option>
            ))}
          </select>
        </div>
      </div>
      
      {/* Tab navigation */}
      <div className="mb-6 flex justify-center space-x-2">
        <button 
          onClick={() => setActiveTab('overview')} 
          className={`px-4 py-2 rounded ${activeTab === 'overview' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
        >
          Overview
        </button>
        <button 
          onClick={() => setActiveTab('regional')} 
          className={`px-4 py-2 rounded ${activeTab === 'regional' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
        >
          Regional Analysis
        </button>
        <button 
          onClick={() => setActiveTab('health')} 
          className={`px-4 py-2 rounded ${activeTab === 'health' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
        >
          Health Impact
        </button>
        <button 
          onClick={() => setActiveTab('business')} 
          className={`px-4 py-2 rounded ${activeTab === 'business' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
        >
          Business Impact
        </button>
      </div>
      
      {/* Main content area */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {activeTab === 'overview' && (
          <>
            {/* Regional Statistics Card */}
            <div className="bg-white p-4 rounded-lg shadow-md">
              <h2 className="text-xl font-semibold mb-4">Regional Statistics</h2>
              <div className="overflow-x-auto">
                <table className="min-w-full">
                  <thead>
                    <tr className="bg-gray-100">
                      <th className="py-2 px-4 text-left">Region</th>
                      <th className="py-2 px-4 text-right">Avg Consumption</th>
                      <th className="py-2 px-4 text-right">Avg Temperature</th>
                      <th className="py-2 px-4 text-right">Avg Renewable %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(regionStats).map(([region, stats]) => (
                      <tr key={region} className="border-t">
                        <td className="py-2 px-4">{region}</td>
                        <td className="py-2 px-4 text-right">{stats.avgConsumption.toFixed(2)}</td>
                        <td className="py-2 px-4 text-right">{stats.avgTemperature.toFixed(2)}°F</td>
                        <td className="py-2 px-4 text-right">{stats.avgRenewable.toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            
            {/* Renewable Energy Business Improvement */}
            <div className="bg-white p-4 rounded-lg shadow-md">
              <h2 className="text-xl font-semibold mb-4">Renewable Energy Business Improvement</h2>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={getRenewableBusinessImpactData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="region" />
                  <YAxis />
                  <Tooltip formatter={(value) => value.toFixed(2)} />
                  <Legend />
                  <Bar dataKey="percentImprovement" name="% Business Activity Improvement" fill="#82ca9d" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            
            {/* Consumption vs Business Correlation */}
            <div className="bg-white p-4 rounded-lg shadow-md">
              <h2 className="text-xl font-semibold mb-4">Consumption vs Business Correlation</h2>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart 
                  outerRadius={90} 
                  width={500} 
                  height={300} 
                  data={getCorrelationData()}
                >
                  <PolarGrid />
                  <PolarAngleAxis dataKey="region" />
                  <PolarRadiusAxis angle={30} domain={[0, 1]} />
                  <Radar 
                    name="Consumption-Business Correlation" 
                    dataKey="Consumption vs Business" 
                    stroke="#8884d8" 
                    fill="#8884d8" 
                    fillOpacity={0.6} 
                  />
                  <Tooltip formatter={(value) => value.toFixed(4)} />
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            </div>
            </div>
            
            {/* Daily Consumption Trends */}
            <div className="bg-white p-4 rounded-lg shadow-md">
              <h2 className="text-xl font-semibold mb-4">Daily Consumption Trends</h2>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={getDailyConsumptionData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  {regions.map((region, i) => (
                    <Line 
                      key={region}
                      type="monotone" 
                      dataKey="avgConsumption" 
                      data={getDailyConsumptionData().filter(d => d.region === region)} 
                      name={region} 
                      stroke={COLORS[i % COLORS.length]} 
                      strokeWidth={selectedRegion === region ? 3 : 1}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
            
            {/* Correlations Chart */}
            <div className="bg-white p-4 rounded-lg shadow-md">
              <h2 className="text-xl font-semibold mb-4">Key Correlations</h2>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={getCorrelationData()} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[-1, 1]} />
                  <YAxis dataKey="region" type="category" width={80} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="Consumption vs Temperature" fill="#8884d8" />
                  <Bar dataKey="Consumption vs Business" fill="#82ca9d" />
                  <Bar dataKey="Temperature vs Health" fill="#ffc658" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            
            {/* Health Impact Summary */}
            <div className="bg-white p-4 rounded-lg shadow-md">
              <h2 className="text-xl font-semibold mb-4">Health Impact of Energy Peaks</h2>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={getHealthImpactData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="region" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="regularAdmissions" name="Regular Admissions" fill="#8884d8" />
                  <Bar dataKey="peakAdmissions" name="After Energy Peaks" fill="#82ca9d" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        )}
        
        {activeTab === 'regional' && selectedRegion && (
          <>
            {/* Regional Consumption by Hour and Day */}
            <div className="bg-white p-4 rounded-lg shadow-md">
              <h2 className="text-xl font-semibold mb-4">{selectedRegion}: Consumption Patterns</h2>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={getHourlyPatternData(selectedRegion)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="hour" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="avgConsumption" name="Avg Consumption" fill="#8884d8">
                    {getHourlyPatternData(selectedRegion).map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={CONSUMPTION_COLORS[Math.floor(entry.hour / 6)]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            
            {/* Consumption vs Temperature Scatter */}
            <div className="bg-white p-4 rounded-lg shadow-md">
              <h2 className="text-xl font-semibold mb-4">{selectedRegion}: Consumption vs Temperature</h2>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" dataKey="temperature" name="Temperature" unit="°F" />
                  <YAxis type="number" dataKey="consumption" name="Consumption" />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Scatter 
                    name="Consumption vs Temperature" 
                    data={getConsumptionTemperatureData(selectedRegion)}
                    fill="#8884d8" 
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
            
            {/* Region Stats Card */}
            <div className="bg-white p-4 rounded-lg shadow-md col-span-1 md:col-span-2">
              <h2 className="text-xl font-semibold mb-4">{selectedRegion}: Detailed Statistics</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-blue-50 p-4 rounded-lg text-center">
                  <h3 className="text-lg font-medium text-blue-800">Average Consumption</h3>
                  <p className="text-3xl font-bold text-blue-600">
                    {regionStats[selectedRegion]?.avgConsumption.toFixed(2)}
                  </p>
                  <p className="text-sm text-gray-600">kWh</p>
                </div>
                <div className="bg-green-50 p-4 rounded-lg text-center">
                  <h3 className="text-lg font-medium text-green-800">Renewable Energy</h3>
                  <p className="text-3xl font-bold text-green-600">
                    {regionStats[selectedRegion]?.avgRenewable.toFixed(2)}%
                  </p>
                  <p className="text-sm text-gray-600">of total generation</p>
                </div>
                <div className="bg-yellow-50 p-4 rounded-lg text-center">
                  <h3 className="text-lg font-medium text-yellow-800">Peak Consumption</h3>
                  <p className="text-3xl font-bold text-yellow-600">
                    {regionStats[selectedRegion]?.maxConsumption.toFixed(2)}
                  </p>
                  <p className="text-sm text-gray-600">kWh (maximum)</p>
                </div>
                <div className="bg-red-50 p-4 rounded-lg text-center">
                  <h3 className="text-lg font-medium text-red-800">Health Impact</h3>
                  <p className="text-3xl font-bold text-red-600">
                    {peakAnalysis[selectedRegion]?.percentDifference.toFixed(2)}%
                  </p>
                  <p className="text-sm text-gray-600">increase in admissions after peaks</p>
                </div>
              </div>
            </div>
          </>
        )}
        
        {activeTab === 'health' && (
          <>
            {/* Health Impact Analysis */}
            <div className="bg-white p-4 rounded-lg shadow-md col-span-1 md:col-span-2">
              <h2 className="text-xl font-semibold mb-4">Health Impact of Energy Consumption Patterns</h2>
              <p className="mb-4">
                Analysis shows relationships between energy consumption patterns and public health metrics. 
                Peaks in energy consumption are often associated with increased hospital admissions, with differences by region.
              </p>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={getHealthImpactData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="region" />
                  <YAxis />
                  <Tooltip formatter={(value) => value.toFixed(2)} />
                  <Legend />
                  <Bar dataKey="percentIncrease" name="% Increase in Hospital Admissions" fill="#FF8042">
                    {getHealthImpactData().map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={HEALTH_COLORS[Math.min(3, Math.floor(entry.percentIncrease / 5))]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            
            {/* Temperature vs Health Correlation */}
            <div className="bg-white p-4 rounded-lg shadow-md">
              <h2 className="text-xl font-semibold mb-4">Temperature vs Health Correlation</h2>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={getCorrelationData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="region" />
                  <YAxis domain={[-1, 1]} />
                  <Tooltip formatter={(value) => value.toFixed(4)} />
                  <Legend />
                  <Bar dataKey="Temperature vs Health" name="Temperature-Health Correlation" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            
            {/* Consumption vs Health Correlation */}
            <div className="bg-white p-4 rounded-lg shadow-md">
              <h2 className="text-xl font-semibold mb-4">Consumption vs Health Correlation</h2>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={getCorrelationData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="region" />
                  <YAxis domain={[-1, 1]} />
                  <Tooltip formatter={(value) => value.toFixed(4)} />
                  <Legend />
                  <Bar dataKey="Consumption vs Health" name="Consumption-Health Correlation" fill="#82ca9d" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        )}
        
        {activeTab === 'business' && (
          <>
            {/* Business Impact Analysis */}
            <div className="bg-white p-4 rounded-lg shadow-md col-span-1 md:col-span-2">
              <h2 className="text-xl font-semibold mb-4">Business Impact of Renewable Energy</h2>
              <p className="mb-4">
                Higher renewable energy percentages correlate with improved business activity metrics.
                This analysis quantifies the business benefits of renewable energy adoption.
              </p>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={getRenewableBusinessImpactData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="region" />
                  <YAxis />
                  <Tooltip formatter={(value) => value.toFixed(2)} />
                  <Legend />
                  <Bar dataKey="lowRenewableBusiness" name="Business Activity (Low Renewable %)" fill="#8884d8" />
                  <Bar dataKey="highRenewableBusiness" name="Business Activity (High Renewable %)" fill="#82ca9d" />
                </BarChart>
              </ResponsiveContainer>
            </div>