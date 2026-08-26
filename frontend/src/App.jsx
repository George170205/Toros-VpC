import React, { useState, useEffect, useRef } from 'react'
import { 
  Camera, Users, Settings, Activity, ShieldAlert, 
  Trash2, RefreshCw, CheckCircle2, UserCheck, AlertTriangle
} from 'lucide-react'

export default function App() {
  const [isConnected, setIsConnected] = useState(false)
  const [frameSrc, setFrameSrc] = useState('')
  const [events, setEvents] = useState([])
  const [persons, setPersons] = useState([])
  const [activeTab, setActiveTab] = useState('events')
  const [config, setConfig] = useState({
    min_quality: 0.45,
    min_verification_score: 0.68,
    min_margin: 0.08,
    min_consistency: 0.72
  })

  const wsRef = useRef(null)

  // 1. Cargar datos iniciales desde la API REST
  const fetchInitialData = async () => {
    try {
      const configRes = await fetch('http://localhost:8000/api/config')
      if (configRes.ok) {
        const configData = await configRes.json()
        setConfig(configData)
      }

      const personsRes = await fetch('http://localhost:8000/api/persons')
      if (personsRes.ok) {
        const personsData = await personsRes.json()
        setPersons(personsData)
      }

      const visitsRes = await fetch('http://localhost:8000/api/visits')
      if (visitsRes.ok) {
        const visitsData = await visitsRes.json()
        const mappedEvents = visitsData.map(v => ({
          session_id: v.visit_id,
          track_id: v.track_id,
          camera_id: v.camera_id,
          decision: "MATCH",
          person_id: v.person_id,
          public_code: v.public_code,
          confidence: v.confidence,
          timestamp: v.timestamp,
          reason: "Registro Histórico"
        }))
        setEvents(mappedEvents)
      }
    } catch (err) {
      console.error("Error al cargar datos iniciales:", err)
    }
  }

  useEffect(() => {
    fetchInitialData()

    // 2. Conectar al WebSocket de streaming
    const connectWebSocket = () => {
      const wsUrl = 'ws://localhost:8000/api/ws/stream'
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        setIsConnected(true)
        console.log("WebSocket conectado.")
        // Enviar pings regulares para mantener vivo
        const pingInterval = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send("ping")
          }
        }, 10000)
        ws._pingInterval = pingInterval
      }

      ws.onmessage = (event) => {
        if (event.data === "pong") return
        
        try {
          const msg = JSON.parse(event.data)
          if (msg.type === "frame") {
            setFrameSrc("data:image/jpeg;base64," + msg.data)
          } else if (msg.type === "event") {
            setEvents(prev => [msg.data, ...prev.slice(0, 29)])
            // Refrescar lista de personas si hay nueva inscripción
            if (msg.data.decision === "UNKNOWN" || msg.data.decision === "MATCH") {
              fetch('http://localhost:8000/api/persons')
                .then(res => res.ok && res.json())
                .then(data => data && setPersons(data))
            }
          }
        } catch (err) {
          console.error("Error al procesar mensaje de WebSocket:", err)
        }
      }

      ws.onclose = () => {
        setIsConnected(false)
        console.log("WebSocket cerrado. Reintentando en 3s...")
        clearInterval(ws._pingInterval)
        setTimeout(connectWebSocket, 3000)
      }

      ws.onerror = (err) => {
        console.error("WebSocket error:", err)
        ws.close()
      }
    }

    connectWebSocket()

    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  // 3. Modificar sliders y guardar configuración en la BD
  const handleConfigChange = (key, value) => {
    setConfig(prev => ({
      ...prev,
      [key]: parseFloat(value)
    }))
  }

  const saveConfigToBackend = async (updatedConfig) => {
    try {
      await fetch('http://localhost:8000/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: json_stringify_safe(updatedConfig)
      })
    } catch (err) {
      console.error("Error al guardar configuración:", err)
    }
  }

  const json_stringify_safe = (obj) => {
    // Evitar serializar campos del backend que no correspondan
    const { configuration_id, version, created_at, active, ...params } = obj
    return JSON.stringify(params)
  }

  // 4. Limpiar Base de Datos
  const handleResetDatabase = async () => {
    if (confirm("¿Estás seguro de que deseas borrar toda la base de datos? Esto eliminará todos los visitantes registrados y su historial.")) {
      try {
        const res = await fetch('http://localhost:8000/api/reset', { method: 'POST' })
        if (res.ok) {
          setEvents([])
          setPersons([])
          alert("Base de datos reseteada con éxito.")
        }
      } catch (err) {
        alert("Error al resetear base de datos: " + err)
      }
    }
  }

  const formatTime = (isoString) => {
    try {
      const d = new Date(isoString)
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    } catch (e) {
      return ""
    }
  }

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header glass">
        <div className="brand-section">
          <span className="brand-icon">🏟️</span>
          <h1 className="brand-title">Toros Recognition</h1>
        </div>
        <div className="brand-section">
          <button className="reset-btn" onClick={handleResetDatabase}>
            <Trash2 size={14} style={{ marginRight: '6px', verticalAlign: 'middle' }} />
            Reset BD
          </button>
          <div className="status-badge">
            <span className={`status-indicator ${isConnected ? 'online' : 'offline'}`}></span>
            {isConnected ? 'ONLINE' : 'OFFLINE'}
          </div>
        </div>
      </header>

      {/* Grid Principal */}
      <main className="dashboard-grid">
        
        {/* Panel Izquierdo: Streaming en vivo y Sliders */}
        <div className="left-panel">
          
          {/* Card de Video */}
          <div className="video-card glass">
            <div className="video-header">
              <h2 className="video-title">
                <Camera size={18} color="var(--primary)" />
                Stream de Acceso en Vivo
              </h2>
              {frameSrc && (
                <span className="status-badge" style={{ padding: '2px 8px', fontSize: '11px', color: 'var(--success)' }}>
                  STREAMING
                </span>
              )}
            </div>
            <div className="video-container">
              {frameSrc ? (
                <img src={frameSrc} alt="Stream en vivo" className="video-feed" />
              ) : (
                <div className="video-placeholder">
                  <div className="spinner"></div>
                  <p>Esperando señal de cámara...</p>
                </div>
              )}
            </div>
          </div>

          {/* Card de Configuración de Thresholds */}
          <div className="config-card glass">
            <h3 className="video-title" style={{ marginBottom: '12px' }}>
              <Settings size={18} color="var(--primary)" />
              Calibración de Parámetros en Caliente
            </h3>
            
            <div className="config-grid">
              
              <div className="config-item">
                <label className="config-label">
                  <span>Calidad de Rostro Mínima</span>
                  <span style={{ fontWeight: '700' }}>{config.min_quality.toFixed(2)}</span>
                </label>
                <input 
                  type="range" min="0.1" max="0.9" step="0.05"
                  value={config.min_quality}
                  onChange={(e) => handleConfigChange('min_quality', e.target.value)}
                  onMouseUp={() => saveConfigToBackend(config)}
                  className="config-slider" 
                />
              </div>

              <div className="config-item">
                <label className="config-label">
                  <span>Verificación Mínima Match</span>
                  <span style={{ fontWeight: '700' }}>{config.min_verification_score.toFixed(2)}</span>
                </label>
                <input 
                  type="range" min="0.4" max="0.95" step="0.02"
                  value={config.min_verification_score}
                  onChange={(e) => handleConfigChange('min_verification_score', e.target.value)}
                  onMouseUp={() => saveConfigToBackend(config)}
                  className="config-slider" 
                />
              </div>

              <div className="config-item">
                <label className="config-label">
                  <span>Margen de Decisión</span>
                  <span style={{ fontWeight: '700' }}>{config.min_margin.toFixed(2)}</span>
                </label>
                <input 
                  type="range" min="0.02" max="0.25" step="0.01"
                  value={config.min_margin}
                  onChange={(e) => handleConfigChange('min_margin', e.target.value)}
                  onMouseUp={() => saveConfigToBackend(config)}
                  className="config-slider" 
                />
              </div>

            </div>
          </div>

        </div>

        {/* Panel Derecho: Historial de decisiones y Personas registradas */}
        <div className="right-panel glass">
          <div className="panel-tab-header">
            <button 
              className={`tab-btn ${activeTab === 'events' ? 'active' : ''}`}
              onClick={() => setActiveTab('events')}
            >
              <Activity size={16} />
              Ingresos en Vivo
            </button>
            <button 
              className={`tab-btn ${activeTab === 'persons' ? 'active' : ''}`}
              onClick={() => setActiveTab('persons')}
            >
              <Users size={16} />
              Personas Registradas
            </button>
          </div>

          <div className="panel-content">
            
            {/* Contenido Pestaña 1: Ingresos en Vivo */}
            {activeTab === 'events' && (
              <div className="event-list">
                {events.length === 0 ? (
                  <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
                    <CheckCircle2 size={32} style={{ marginBottom: '12px', opacity: '0.4' }} />
                    <p>No hay eventos de ingreso registrados aún.</p>
                  </div>
                ) : (
                  events.map((evt, idx) => (
                    <div className="event-card" key={evt.session_id + "_" + idx}>
                      <div className="event-info">
                        <div className={`event-avatar ${evt.decision.toLowerCase()}`}>
                          {evt.decision === "MATCH" && <UserCheck size={20} color="var(--success)" />}
                          {evt.decision === "UNKNOWN" && <Users size={20} color="var(--primary)" />}
                          {evt.decision === "AMBIGUOUS" && <AlertTriangle size={20} color="var(--warning)" />}
                        </div>
                        <div className="event-details">
                          <span className="event-user-code">{evt.public_code}</span>
                          <span className="event-meta">
                            Confianza: {(evt.confidence * 100).toFixed(1)}% | {formatTime(evt.timestamp)}
                          </span>
                        </div>
                      </div>
                      <span className={`event-status-pill ${evt.decision.toLowerCase()}`}>
                        {evt.decision}
                      </span>
                    </div>
                  ))
                )}
              </div>
            )}

            {/* Contenido Pestaña 2: Personas Registradas */}
            {activeTab === 'persons' && (
              <div style={{ overflowX: 'auto' }}>
                {persons.length === 0 ? (
                  <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
                    <Users size={32} style={{ marginBottom: '12px', opacity: '0.4' }} />
                    <p>No hay personas enroladas en la base de datos.</p>
                  </div>
                ) : (
                  <table className="people-table">
                    <thead>
                      <tr>
                        <th>Código Público</th>
                        <th>Estado</th>
                        <th>Visitas</th>
                        <th>Último Ingreso</th>
                        <th>Calidad Prom.</th>
                      </tr>
                    </thead>
                    <tbody>
                      {persons.map(p => (
                        <tr key={p.person_id}>
                          <td style={{ fontWeight: '700', fontFamily: 'var(--font-display)', color: 'var(--primary)' }}>
                            {p.public_code}
                          </td>
                          <td>
                            <span style={{ 
                              fontSize: '10px', 
                              fontWeight: '700', 
                              color: p.status === 'ACTIVE' ? 'var(--success)' : 'var(--text-muted)'
                            }}>
                              {p.status}
                            </span>
                          </td>
                          <td style={{ fontWeight: '600' }}>{p.total_visits}</td>
                          <td style={{ color: 'var(--text-muted)' }}>{formatTime(p.last_seen)}</td>
                          <td style={{ fontWeight: '600' }}>
                            {p.primary_quality ? (p.primary_quality * 100).toFixed(1) + "%" : "-"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            )}

          </div>
        </div>

      </main>
    </div>
  )
}
