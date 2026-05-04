import { useState } from 'react';
import axios from 'axios';
import { Newspaper, ChevronDown, ChevronUp, AlertCircle, Loader2 } from 'lucide-react';

function App() {
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [showUnderTheHood, setShowUnderTheHood] = useState(false);

  const handleClassify = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text before analyzing.');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await axios.post('http://localhost:8000/api/predict', {
        text: inputText
      });
      setResult(response.data);
    } catch (err) {
      setError('Failed to reach the API. Make sure the FastAPI backend is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto space-y-8">
        
        {/* Header Section */}
        <div className="text-center space-y-4">
          <div className="flex justify-center">
            <div className="p-4 bg-blue-100 rounded-full">
              <Newspaper className="w-12 h-12 text-blue-600" />
            </div>
          </div>
          <h1 className="text-4xl font-extrabold text-gray-900 tracking-tight sm:text-5xl">
            AI News Classifier
          </h1>
          <p className="text-lg text-gray-500 max-w-2xl mx-auto">
            Instantly route news articles to the correct department using Deep Learning.
          </p>
        </div>

        {/* Input Section */}
        <div className="bg-white rounded-2xl shadow-xl border border-gray-100 p-6 md:p-8 space-y-6">
          <div className="space-y-2">
            <label htmlFor="article" className="block text-sm font-medium text-gray-700">
              Paste an article headline or snippet below:
            </label>
            <textarea
              id="article"
              rows={5}
              className="block w-full rounded-xl border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-base border p-4 transition duration-150 ease-in-out hover:border-blue-300"
              placeholder="e.g., Apple announces a new high-speed quantum computer chip that will revolutionize the industry..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
            />
          </div>

          {error && (
            <div className="rounded-lg bg-red-50 p-4 flex items-center gap-3 text-red-700">
              <AlertCircle className="h-5 w-5" />
              <p className="text-sm font-medium">{error}</p>
            </div>
          )}

          <button
            onClick={handleClassify}
            disabled={loading}
            className="w-full flex items-center justify-center py-4 px-4 border border-transparent rounded-xl shadow-sm text-lg font-semibold text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
          >
            {loading ? (
              <>
                <Loader2 className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" />
                Analyzing Text...
              </>
            ) : (
              '🔮 Analyze Text'
            )}
          </button>
        </div>

        {/* Results Section */}
        {result && (
          <div className="space-y-6 animate-in fade-in duration-500">
            {/* Winning Category Badge */}
            <div className="bg-white rounded-2xl shadow-lg border border-green-100 p-8 text-center bg-gradient-to-br from-green-50 to-emerald-50">
              <h2 className="text-sm font-semibold text-green-800 uppercase tracking-wide">
                Classification Result
              </h2>
              <div className="mt-2 flex items-center justify-center">
                <span className="text-4xl font-black text-green-600">
                  {result.winning_category}
                </span>
              </div>
            </div>

            {/* Confidence Breakdown */}
            <div className="bg-white rounded-2xl shadow-md border border-gray-100 p-6 md:p-8">
              <h3 className="text-lg font-bold text-gray-900 mb-6 flex items-center gap-2">
                📊 Neural Network Confidence Breakdown
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {Object.entries(result.confidence_breakdown).map(([category, prob]) => {
                  const percentage = (prob * 100).toFixed(2);
                  return (
                    <div key={category} className="space-y-2">
                      <div className="flex justify-between items-center text-sm font-medium text-gray-700">
                        <span>{category}</span>
                        <span>{percentage}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2.5 overflow-hidden">
                        <div
                          className="bg-blue-600 h-2.5 rounded-full transition-all duration-1000 ease-out"
                          style={{ width: `${percentage}%` }}
                        ></div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Under the Hood Accordion */}
            <div className="bg-white rounded-2xl shadow-md border border-gray-100 overflow-hidden">
              <button
                onClick={() => setShowUnderTheHood(!showUnderTheHood)}
                className="w-full flex items-center justify-between p-6 bg-gray-50 hover:bg-gray-100 transition-colors duration-150 focus:outline-none"
              >
                <div className="flex items-center gap-2">
                  <span className="text-lg">🛠️</span>
                  <h3 className="text-md font-bold text-gray-800">
                    See Under the Hood: What did the AI actually read?
                  </h3>
                </div>
                {showUnderTheHood ? (
                  <ChevronUp className="h-5 w-5 text-gray-500" />
                ) : (
                  <ChevronDown className="h-5 w-5 text-gray-500" />
                )}
              </button>
              
              {showUnderTheHood && (
                <div className="p-6 border-t border-gray-100 space-y-6 bg-white animate-in slide-in-from-top-2">
                  <p className="text-sm text-gray-600">
                    Neural networks don't read English the way we do. First, we strip away punctuation, remove filler "stop words" (like "the", "is", "at"), and reduce words to their base roots (lemmatization).
                  </p>
                  
                  <div className="space-y-4">
                    <div>
                      <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">
                        Original Text
                      </h4>
                      <p className="text-sm text-gray-800 italic bg-gray-50 p-3 rounded-lg border border-gray-100">
                        {inputText}
                      </p>
                    </div>

                    <div>
                      <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">
                        What the AI Saw (Cleaned & Lemmatized)
                      </h4>
                      <div className="bg-gray-900 text-gray-100 p-4 rounded-lg font-mono text-sm overflow-x-auto">
                        {result.cleaned_string || "No valid words found."}
                      </div>
                    </div>

                    <div>
                      <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">
                        How it was fed to the Neural Network (Integer Sequence)
                      </h4>
                      <div className="bg-gray-900 text-gray-100 p-4 rounded-lg font-mono text-sm overflow-x-auto break-all">
                        [{result.padded_sequence.join(', ')}]
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
