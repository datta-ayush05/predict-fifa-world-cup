import React, { useState, useEffect } from 'react';
import './index.css';
import predictionData from './data/predictions.json';

// Utility to calculate Poisson match probabilities
function calculateMatchProbs(lambdaH, lambdaA) {
  let win = 0.0;
  let draw = 0.0;
  let loss = 0.0;
  
  const fact = (n) => (n === 0 || n === 1) ? 1 : n * fact(n - 1);
  
  for (let i = 0; i < 10; i++) {
    for (let j = 0; j < 10; j++) {
      const prob = (Math.exp(-lambdaH) * Math.pow(lambdaH, i) / fact(i)) *
                   (Math.exp(-lambdaA) * Math.pow(lambdaA, j) / fact(j));
      if (i > j) win += prob;
      else if (i === j) draw += prob;
      else loss += prob;
    }
  }
  
  const total = win + draw + loss;
  return {
    win: win / total,
    draw: draw / total,
    loss: loss / total
  };
}

function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [teams, setTeams] = useState([]);
  const [selectedTeam, setSelectedTeam] = useState('');
  const [selectedTeam2, setSelectedTeam2] = useState('');
  const [teamData, setTeamData] = useState(null);
  const [globalStats, setGlobalStats] = useState([]);

  // Load static data on mount
  useEffect(() => {
    try {
      const teamsObj = predictionData.teams || {};
      const teamNames = Object.keys(teamsObj).sort();
      setTeams(teamNames);

      const teamList = teamNames.map(name => ({
        name: name,
        win_prob: teamsObj[name].win_prob || 0,
        final_prob: teamsObj[name].final_prob || 0,
        sf_prob: teamsObj[name].sf_prob || 0,
      }));
      
      teamList.sort((a, b) => b.win_prob - a.win_prob);
      setGlobalStats(teamList);
    } catch (err) {
      console.error("Error loading static data:", err);
    }
  }, []);

  // Update specific team stats when selected
  useEffect(() => {
    if (!selectedTeam) {
      setTeamData(null);
      setSelectedTeam2('');
      return;
    }

    const teamsObj = predictionData.teams || {};
    const probMatrix = predictionData.prob_matrix || {};
    const stats = teamsObj[selectedTeam];
    
    if (!stats) return;

    const group = stats.group;
    const matches = [];
    let customMatch = null;

    // Head-to-Head logic for selectedTeam2
    if (selectedTeam2 && selectedTeam2 !== selectedTeam) {
      const key1 = `${selectedTeam}::${selectedTeam2}`;
      const key2 = `${selectedTeam2}::${selectedTeam}`;
      let probs = null;
      let xgFor = 0;
      let xgAgainst = 0;
      
      if (probMatrix[key1]) {
        probs = probMatrix[key1];
        xgFor = probs[0];
        xgAgainst = probs[1];
      } else if (probMatrix[key2]) {
        probs = probMatrix[key2];
        xgFor = probs[1];
        xgAgainst = probs[0];
      }
      
      if (probs) {
        const matchP = calculateMatchProbs(xgFor, xgAgainst);
        customMatch = {
          opponent: selectedTeam2,
          xG_for: xgFor,
          xG_against: xgAgainst,
          win_prob: matchP.win,
          draw_prob: matchP.draw,
          loss_prob: matchP.loss
        };
      }
    } else {
      // Get probabilities against teams in same group
      Object.keys(teamsObj).forEach(t2 => {
        const t2Stats = teamsObj[t2];
        if (t2 !== selectedTeam && t2Stats.group === group) {
          const key1 = `${selectedTeam}::${t2}`;
          const key2 = `${t2}::${selectedTeam}`;
          
          let probs = null;
          let xgFor = 0;
          let xgAgainst = 0;
          
          if (probMatrix[key1]) {
            probs = probMatrix[key1];
            xgFor = probs[0];
            xgAgainst = probs[1];
          } else if (probMatrix[key2]) {
            probs = probMatrix[key2];
            xgFor = probs[1]; // reversed
            xgAgainst = probs[0];
          }
          
          if (probs) {
            const matchP = calculateMatchProbs(xgFor, xgAgainst);
            matches.push({
              opponent: t2,
              xG_for: xgFor,
              xG_against: xgAgainst,
              win_prob: matchP.win,
              draw_prob: matchP.draw,
              loss_prob: matchP.loss
            });
          }
        }
      });
    }

    setTeamData({
      stats: stats,
      group_matches: matches,
      custom_match: customMatch
    });

  }, [selectedTeam, selectedTeam2]);

  return (
    <div className="app-container">
      <header className="header" style={{paddingBottom: '0'}}>
        <h1>FIFA World Cup 2026 Predictions</h1>
        <p>AI-powered Monte Carlo Simulation Analytics</p>
        
        <nav className="nav-tabs" style={{marginTop: '2rem', display: 'flex', justifyContent: 'center', gap: '1rem'}}>
          <button 
            className={`tab-button ${activeTab === 'home' ? 'active' : ''}`}
            onClick={() => setActiveTab('home')}
            style={{
              padding: '0.8rem 1.5rem', 
              background: activeTab === 'home' ? 'var(--accent)' : 'transparent',
              color: activeTab === 'home' ? '#fff' : 'var(--text-color)',
              border: '2px solid var(--accent)',
              borderRadius: '30px',
              cursor: 'pointer',
              fontWeight: '600',
              transition: 'all 0.3s',
              fontFamily: 'inherit'
            }}
          >
            Dashboard
          </button>
          <button 
            className={`tab-button ${activeTab === 'about' ? 'active' : ''}`}
            onClick={() => setActiveTab('about')}
            style={{
              padding: '0.8rem 1.5rem', 
              background: activeTab === 'about' ? 'var(--accent)' : 'transparent',
              color: activeTab === 'about' ? '#fff' : 'var(--text-color)',
              border: '2px solid var(--accent)',
              borderRadius: '30px',
              cursor: 'pointer',
              fontWeight: '600',
              transition: 'all 0.3s',
              fontFamily: 'inherit'
            }}
          >
            About Methodology
          </button>
        </nav>
      </header>

      {activeTab === 'home' && (
        <div className="home-view" style={{animation: 'fadeIn 0.5s ease-out'}}>
          <div className="selector-container">
            <select value={selectedTeam} onChange={(e) => setSelectedTeam(e.target.value)}>
              <option value="">-- Select Team 1 --</option>
              {teams.map(t => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
            
            {selectedTeam && (
              <select value={selectedTeam2} onChange={(e) => setSelectedTeam2(e.target.value)} style={{marginLeft: '1rem'}}>
                <option value="">-- Any Opponent (Optional) --</option>
                {teams.filter(t => t !== selectedTeam).map(t => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            )}
          </div>

          {teamData && !selectedTeam2 && (
            <div className="dashboard">
              <div className="card glass">
                <h3>Win Tournament</h3>
                <div className="stat">{(teamData.stats.win_prob * 100).toFixed(1)}%</div>
                <div className="progress-bar-container">
                  <div className="progress-bar" style={{ width: `${Math.min(teamData.stats.win_prob * 100, 100)}%` }}></div>
                </div>
              </div>
              
              <div className="card glass">
                <h3>Reach Final</h3>
                <div className="stat">{(teamData.stats.final_prob * 100).toFixed(1)}%</div>
                <div className="progress-bar-container">
                  <div className="progress-bar" style={{ width: `${Math.min(teamData.stats.final_prob * 100, 100)}%` }}></div>
                </div>
              </div>
              
              <div className="card glass">
                <h3>Advance from Group</h3>
                <div className="stat">{(teamData.stats.group_adv_prob * 100).toFixed(1)}%</div>
                <div className="progress-bar-container">
                  <div className="progress-bar" style={{ width: `${Math.min(teamData.stats.group_adv_prob * 100, 100)}%` }}></div>
                </div>
              </div>

              {teamData.group_matches && teamData.group_matches.length > 0 && (
                <div className="matches-container glass card">
                  <h3>Group {teamData.stats.group} Match Probabilities</h3>
                  <div className="matches-grid">
                    {teamData.group_matches.map((m, idx) => (
                      <div key={idx} className="match-card">
                        <div className="match-opponent">vs {m.opponent}</div>
                        <div className="match-stats">
                          <div className="match-stat-col">
                            <span>Win</span>
                            <span style={{color: 'var(--success)'}}>{(m.win_prob * 100).toFixed(1)}%</span>
                          </div>
                          <div className="match-stat-col">
                            <span>Draw</span>
                            <span style={{color: 'var(--text-muted)'}}>{(m.draw_prob * 100).toFixed(1)}%</span>
                          </div>
                          <div className="match-stat-col">
                            <span>Loss</span>
                            <span style={{color: 'var(--danger)'}}>{(m.loss_prob * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                        <div style={{marginTop: '0.8rem', fontSize: '0.85rem', color: 'var(--text-muted)'}}>
                          Expected Goals: {m.xG_for.toFixed(2)} - {m.xG_against.toFixed(2)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {teamData && selectedTeam2 && teamData.custom_match && (
            <div className="dashboard" style={{justifyContent: 'center', display: 'flex'}}>
              <div className="matches-container glass card" style={{maxWidth: '500px', width: '100%'}}>
                <h3 style={{textAlign: 'center', marginBottom: '2rem'}}>Head-to-Head Prediction</h3>
                <div className="match-card" style={{border: 'none', background: 'transparent', boxShadow: 'none'}}>
                  <div className="match-opponent" style={{fontSize: '1.5rem', marginBottom: '2rem'}}>
                    {selectedTeam} <span style={{color: 'var(--accent)'}}>vs</span> {selectedTeam2}
                  </div>
                  <div className="match-stats" style={{marginBottom: '2rem'}}>
                    <div className="match-stat-col">
                      <span>{selectedTeam} Win</span>
                      <span style={{color: 'var(--success)', fontSize: '2rem'}}>{(teamData.custom_match.win_prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className="match-stat-col">
                      <span>Draw</span>
                      <span style={{color: 'var(--text-muted)', fontSize: '2rem'}}>{(teamData.custom_match.draw_prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className="match-stat-col">
                      <span>{selectedTeam2} Win</span>
                      <span style={{color: 'var(--danger)', fontSize: '2rem'}}>{(teamData.custom_match.loss_prob * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                  <div style={{fontSize: '1rem', color: 'var(--text-muted)'}}>
                    Expected Goals: {teamData.custom_match.xG_for.toFixed(2)} - {teamData.custom_match.xG_against.toFixed(2)}
                  </div>
                </div>
              </div>
            </div>
          )}

          {!selectedTeam && globalStats.length > 0 && (
            <div className="table-container glass card">
              <h3>Global Tournament Predictions</h3>
              <div className="table-wrapper">
                <table>
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Team</th>
                      <th>Win %</th>
                      <th>Final %</th>
                      <th>Semi-Final %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {globalStats.map((team, index) => (
                      <tr key={team.name}>
                        <td>{index + 1}</td>
                        <td style={{fontWeight: '600'}}>{team.name}</td>
                        <td>{(team.win_prob * 100).toFixed(1)}%</td>
                        <td>{(team.final_prob * 100).toFixed(1)}%</td>
                        <td>{(team.sf_prob * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'about' && (
        <div className="about-view glass card" style={{maxWidth: '900px', margin: '2rem auto', padding: '3rem', textAlign: 'left', lineHeight: '1.8', animation: 'fadeIn 0.5s ease-out'}}>
          <h2 style={{color: 'var(--accent)', marginBottom: '1.5rem', fontSize: '2rem'}}>About the Methodology</h2>
          <p style={{marginBottom: '1.5rem', fontSize: '1.1rem'}}>
            This repository presents a sophisticated prediction framework for the FIFA World Cup 2026, leveraging a <strong>Graph Attention Network (GAT)</strong> to forecast match outcomes and tournament progression. By representing international football as a dynamic temporal graph, the model integrates a holistic feature set comprising historical match records and continuous Elo rating dynamics. The system outputs expected goals via a Poisson distribution, enabling statistically rigorous Monte Carlo simulations of the entire tournament structure.
          </p>
          
          <h3 style={{marginTop: '2rem', marginBottom: '1rem', color: 'var(--text-color)'}}>Core Components</h3>
          <ul style={{paddingLeft: '1.5rem', color: 'var(--text-muted)', fontSize: '1.05rem'}}>
            <li style={{marginBottom: '1.2rem'}}>
              <strong style={{color: 'var(--text-color)'}}>Temporal Graph Construction:</strong> Historical national teams constitute the nodes, while past match-ups form the weighted edges. This structure allows the model to learn complex inter-team relationships beyond direct head-to-head records.
            </li>
            <li style={{marginBottom: '1.2rem'}}>
              <strong style={{color: 'var(--text-color)'}}>Continuous Time-Decay Weighting:</strong> To balance historical precedent with current form, the model applies an exponential decay function to past matches. A calibrated 3-year half-life ensures recent performances exert a dominant influence on the learned embeddings.
            </li>
            <li style={{marginBottom: '1.2rem'}}>
              <strong style={{color: 'var(--text-color)'}}>Integrated Elo Rating System:</strong> A bespoke rolling Elo engine provides baseline strength assessments. These ratings dynamically update based on match importance and are embedded directly into the node and edge feature spaces.
            </li>
            <li style={{marginBottom: '1.2rem'}}>
              <strong style={{color: 'var(--text-color)'}}>Poisson Goal Expectancy:</strong> The GAT architecture outputs expected goals for both teams. During simulation, native Poisson sampling drives the generation of match scores and determines knockout phase progressions.
            </li>
          </ul>
        </div>
      )}

      <footer className="footer">
        <div className="photo-frame">
          {/* Your photo loads via CSS background */}
        </div>
        <p>Made by Ayush</p>
      </footer>
    </div>
  );
}

export default App;
