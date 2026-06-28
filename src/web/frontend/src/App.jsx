import React, { useState, useEffect } from 'react';
import './index.css';
import predictionData from './data/predictions.json';
import fixtureData from './data/fixtures.json';

function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [globalStats, setGlobalStats] = useState([]);

  // Load static data on mount
  useEffect(() => {
    try {
      const teamsObj = predictionData.teams || {};
      const teamNames = Object.keys(teamsObj).sort();

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

  // Unused state removed

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
        <div className="matches-view animation-fade-in" style={{animation: 'fadeIn 0.5s ease-out'}}>
          {globalStats.length > 0 && (
            <div className="table-container glass card" style={{marginBottom: '3rem'}}>
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

          {fixtureData.matches && fixtureData.matches.length > 0 && (
            <div className="matches-container">
              <h3 style={{color: 'var(--text-color)', marginBottom: '1.5rem', textAlign: 'center'}}>Tournament Fixtures</h3>
              
              {(() => {
                const groupedFixtures = fixtureData.matches.reduce((acc, m) => {
                  const date = m.date || 'TBD';
                  if (!acc[date]) acc[date] = [];
                  acc[date].push(m);
                  return acc;
                }, {});

                const sortedDates = Object.keys(groupedFixtures).sort((a, b) => {
                  if (a === 'TBD') return 1;
                  if (b === 'TBD') return -1;
                  return a.localeCompare(b);
                });

                return sortedDates.map(date => (
                  <div key={date} style={{marginBottom: '2.5rem', textAlign: 'center'}}>
                    <div style={{
                      padding: '0.5rem 1.5rem', 
                      background: 'rgba(255, 255, 255, 0.05)', 
                      borderRadius: '20px', 
                      marginBottom: '1rem',
                      fontSize: '0.9rem',
                      color: 'var(--text-muted)',
                      display: 'inline-block',
                      border: '1px solid rgba(255, 255, 255, 0.1)'
                    }}>
                      {date === 'TBD' ? 'To Be Determined' : new Date(date).toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
                    </div>
                    
                    <div className="matches-grid" style={{display: 'flex', flexDirection: 'column', gap: '1rem', alignItems: 'center'}}>
                      {groupedFixtures[date].map((m, idx) => (
                        <div key={idx} className="match-card glass" style={{
                          display: 'flex', 
                          flexDirection: 'column', 
                          padding: '1.5rem', 
                          borderRadius: '12px',
                          width: '100%',
                          maxWidth: '600px'
                        }}>
                          <div style={{fontSize: '0.85rem', color: 'var(--accent)', fontWeight: 'bold', marginBottom: '1rem', textTransform: 'uppercase'}}>
                            {m.stage}
                          </div>
                          
                          <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                            {/* Team 1 Side */}
                            <div style={{flex: 1, textAlign: 'right'}}>
                              <div style={{fontSize: '1.3rem', fontWeight: 'bold', color: 'var(--text-main)'}}>
                                {m.team1}
                              </div>
                              <div style={{fontSize: '0.9rem', color: 'var(--text-muted)', marginTop: '0.3rem'}}>
                                xG: <span style={{color: 'var(--text-color)', fontWeight: 'bold'}}>{m.prediction.xG1.toFixed(2)}</span>
                              </div>
                            </div>

                            {/* Center Score */}
                            <div style={{flex: '0 0 100px', textAlign: 'center', fontSize: '1.8rem', fontWeight: 'bold', display: 'flex', justifyContent: 'center', gap: '0.5rem', background: 'var(--glass-border)', padding: '0.5rem', borderRadius: '8px', margin: '0 1.5rem'}}>
                              {m.result.status === 'Finished' ? (
                                <>
                                  <span style={{color: 'var(--text-main)'}}>{m.result.score1}</span>
                                  <span style={{color: 'var(--text-muted)'}}>-</span>
                                  <span style={{color: 'var(--text-main)'}}>{m.result.score2}</span>
                                </>
                              ) : (
                                <span style={{fontSize: '1.2rem', color: 'var(--text-muted)'}}>vs</span>
                              )}
                            </div>

                            {/* Team 2 Side */}
                            <div style={{flex: 1, textAlign: 'left'}}>
                              <div style={{fontSize: '1.3rem', fontWeight: 'bold', color: 'var(--text-main)'}}>
                                {m.team2}
                              </div>
                              <div style={{fontSize: '0.9rem', color: 'var(--text-muted)', marginTop: '0.3rem'}}>
                                xG: <span style={{color: 'var(--text-color)', fontWeight: 'bold'}}>{m.prediction.xG2.toFixed(2)}</span>
                              </div>
                            </div>
                          </div>
                          
                          {/* Prediction Footer */}
                          <div style={{marginTop: '1.5rem', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '1rem', display: 'flex', justifyContent: 'center', gap: '2rem', fontSize: '0.85rem'}}>
                             <div style={{display: 'flex', gap: '0.5rem'}}>
                               <span style={{color: 'var(--text-muted)'}}>{m.team1}:</span>
                               <span style={{color: 'var(--success)'}}>{(m.prediction.win_prob * 100).toFixed(0)}%</span>
                             </div>
                             <div style={{display: 'flex', gap: '0.5rem'}}>
                               <span style={{color: 'var(--text-muted)'}}>Draw:</span>
                               <span>{(m.prediction.draw_prob * 100).toFixed(0)}%</span>
                             </div>
                             <div style={{display: 'flex', gap: '0.5rem'}}>
                               <span style={{color: 'var(--text-muted)'}}>{m.team2}:</span>
                               <span style={{color: 'var(--danger)'}}>{(m.prediction.loss_prob * 100).toFixed(0)}%</span>
                             </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ));
              })()}
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
