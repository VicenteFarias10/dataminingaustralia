    document.addEventListener('DOMContentLoaded', function () {
      let mapa = null;
      let markers = [];
      let chart = null;
      let currentTechnique = '';

      // Definir descripciones para cada técnica
      const techniqueDescriptions = {
        clustering: {
          title: "Clasificación de Zonas Climáticas mediante Clustering",
          description: "Esta técnica utiliza K-Means clustering para identificar diferentes tipos de clima en Australia basándose en variables como temperatura, humedad, lluvia y evaporación.",
          requirements: "Columnas requeridas: MinTemp, MaxTemp, Humidity9am, Humidity3pm, Rainfall, Evaporation, Latitud, Longitud"
        },
        confort: {
          title: "Análisis de Confort para Vivir",
          description: "Analiza qué zonas de Australia no son confortables para vivir basándose en condiciones climáticas extremas.",
          requirements: "Columnas requeridas: MinTemp, MaxTemp, Humidity9am, Humidity3pm, Rainfall, Latitud, Longitud"
        },
        lluvia: {
          title: "Predicción de Lluvia",
          description: "Utiliza machine learning para predecir si lloverá mañana basándose en las condiciones climáticas de hoy.",
          requirements: "Columnas requeridas: MinTemp, MaxTemp, Humidity9am, Humidity3pm, Rainfall, Pressure9am, Pressure3pm, Latitud, Longitud"
        },
        evaporacion: {
          title: "Análisis de Evaporación",
          description: "Analiza cómo ciertas variables climáticas afectan al nivel de evaporación diaria.",
          requirements: "Columnas requeridas: MinTemp, MaxTemp, Humidity9am, Humidity3pm, Rainfall, Evaporation, WindSpeed9am, WindSpeed3pm, Latitud, Longitud"
        },
        asociacion: {
          title: "Análisis de Asociación de Variables",
          description: "Identifica qué combinaciones de condiciones climáticas suelen ocurrir juntas en un mismo día.",
          requirements: "Columnas requeridas: MinTemp, MaxTemp, Humidity9am, Humidity3pm, Rainfall, Evaporation, Latitud, Longitud"
        }
      };

      // Función para inicializar el mapa
      function inicializarMapa() {
        if (mapa) {
          mapa.remove();
        }
        
        // Esperar un poco para que el DOM se actualice
        setTimeout(() => {
          const mapaContainer = document.getElementById('mapa');
          if (mapaContainer) {
            mapa = L.map('mapa').setView([-28, 134], 4); // Australia

            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
              attribution: '© OpenStreetMap contributors'
            }).addTo(mapa);
          }
        }, 100);
      }

      // Manejar clics en las tarjetas de técnicas
      document.querySelectorAll('.technique-card').forEach(card => {
        card.addEventListener('click', function() {
          const selectedTechnique = this.getAttribute('data-technique');
          
          // Remover selección anterior
          document.querySelectorAll('.technique-card').forEach(c => c.classList.remove('selected'));
          
          // Seleccionar la tarjeta actual
          this.classList.add('selected');
          
          if (selectedTechnique) {
            currentTechnique = selectedTechnique;
            const technique = techniqueDescriptions[selectedTechnique];
            
            // Mostrar descripción
            document.getElementById('description').innerHTML = `
              <h3><i class="fas fa-info-circle"></i> ${technique.title}</h3>
              <p><strong>Descripción:</strong> ${technique.description}</p>
              <p><strong>Requisitos:</strong> ${technique.requirements}</p>
            `;
            document.getElementById('description').style.display = 'block';
            
            // Mostrar sección de upload
            document.getElementById('uploadSection').style.display = 'block';
            
            // Ocultar resultados anteriores
            document.getElementById('resultsSection').style.display = 'none';
            
            // Limpiar mapa y gráfico
            limpiarMapa();
            limpiarGrafico();
            
            // Scroll suave a la descripción
            document.getElementById('description').scrollIntoView({ 
              behavior: 'smooth', 
              block: 'start' 
            });
            
            // Verificar estado del botón después de seleccionar técnica
            verificarEstadoBoton();
          }
        });
      });

      function limpiarMapa() {
        if (mapa) {
          markers.forEach(marker => mapa.removeLayer(marker));
          markers = [];
        }
      }

      function limpiarGrafico() {
        if (chart) {
          chart.destroy();
          chart = null;
        }
      }

      function mostrarDatos(data) {
        // Mostrar sección de resultados
        document.getElementById('resultsSection').style.display = 'block';
        
        // Inicializar mapa después de mostrar la sección
        inicializarMapa();
        
        // Esperar a que el mapa se inicialice antes de mostrar datos
        setTimeout(() => {
          limpiarMapa();
          limpiarGrafico();

          // Configurar según la técnica seleccionada
          switch(currentTechnique) {
            case 'clustering':
              mostrarResultadosClustering(data);
              break;
            case 'confort':
              mostrarResultadosConfort(data);
              break;
            case 'lluvia':
              mostrarResultadosLluvia(data);
              break;
            case 'evaporacion':
              mostrarResultadosEvaporacion(data);
              break;
            case 'asociacion':
              mostrarResultadosAsociacion(data);
              break;
          }
          
          // Scroll suave a los resultados
          document.getElementById('resultsSection').scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
          });
        }, 200);
      }

      function mostrarResultadosClustering(data) {
        let conteoClusters = { 0: 0, 1: 0 };

        data.forEach(punto => {
          if ((punto.cluster === 0 || punto.cluster === 1) &&
            punto.latitude !== undefined && punto.longitude !== undefined) {

            let color = punto.cluster === 0 ? 'blue' : 'green';
            conteoClusters[punto.cluster]++;

            let marker = L.circleMarker([punto.latitude, punto.longitude], {
              radius: 6,
              fillColor: color,
              fillOpacity: 0.7,
              color: '#000',
              weight: 1
            })
              .bindPopup(`<strong>${punto.Location}</strong><br>Clúster: ${punto.cluster}`)
              .addTo(mapa);

            markers.push(marker);
          }
        });

        // Mostrar información de climas
        document.getElementById('clima-info').innerHTML = `
          <div class="clima-box">
            <div><span class="clima-color" style="background: blue;"></span><strong>Clima Tipo 0</strong></div>
            <p>Este tipo de clima se caracteriza por temperaturas moderadas y menor humedad. Suele presentarse en regiones interiores con menor precipitación anual.</p>
          </div>
          <div class="clima-box">
            <div><span class="clima-color" style="background: green;"></span><strong>Clima Tipo 1</strong></div>
            <p>Este clima corresponde a zonas con mayor humedad y temperaturas variables, típicas de áreas costeras o con influencia marítima.</p>
          </div>
        `;

        // Mostrar descripción de resultados
        document.getElementById('resultDescription').innerHTML = `
          <i class="fas fa-chart-bar"></i> En el gráfico inferior se resumen los resultados del análisis de clustering aplicado a los datos cargados. 
          Las barras representan la cantidad de localidades que fueron clasificadas en uno de los dos tipos de climas principales.
          <br><br>
          El <strong>Clima Tipo 0</strong> suele asociarse a zonas más secas e interiores, con temperaturas moderadas y menor humedad. 
          Por otro lado, el <strong>Clima Tipo 1</strong> se identifica comúnmente en regiones costeras o con mayor influencia marítima, 
          caracterizadas por mayor humedad y variabilidad térmica.
        `;

        // Crear gráfico
        const ctx = document.getElementById('grafico').getContext('2d');
        chart = new Chart(ctx, {
          type: 'bar',
          data: {
            labels: ['Clima Tipo 0', 'Clima Tipo 1'],
            datasets: [{
              label: 'Número de localidades',
              data: [conteoClusters[0], conteoClusters[1]],
              backgroundColor: ['blue', 'green']
            }]
          },
          options: {
            responsive: true,
            plugins: {
              legend: { display: false },
              title: { display: true, text: 'Distribución de localidades por tipo de clima' }
            },
            scales: {
              y: { beginAtZero: true, precision: 0 }
            }
          }
        });
      }

      // Funciones placeholder para las otras técnicas
        function mostrarResultadosConfort(data) {
    // Contar etiquetas de confort
    let confort_stats = {};
    data.forEach(punto => {
      if (punto.etiqueta_confort) {
        if (confort_stats[punto.etiqueta_confort]) {
          confort_stats[punto.etiqueta_confort]++;
        } else {
          confort_stats[punto.etiqueta_confort] = 1;
        }
      }
    });

    // Mostrar información de confort
    let confortHTML = '';
    Object.keys(confort_stats).forEach(etiqueta => {
      const color = etiqueta === 'CONFORME' ? '#27ae60' : '#e74c3c';
      const icon = etiqueta === 'CONFORME' ? 'fa-check-circle' : 'fa-exclamation-triangle';
      
      confortHTML += `
        <div class="clima-box">
          <div><span class="clima-color" style="background: ${color};"></span><strong><i class="fas ${icon}"></i> ${etiqueta}</strong></div>
          <p>Zonas: ${confort_stats[etiqueta]} localidades</p>
        </div>
      `;
    });

    document.getElementById('clima-info').innerHTML = confortHTML;

    // Mostrar marcadores en el mapa
    data.forEach(punto => {
      if (punto.latitude !== undefined && punto.longitude !== undefined) {
        let color = punto.etiqueta_confort === 'CONFORME' ? '#27ae60' : '#e74c3c';
        let radius = punto.confort_score > 70 ? 8 : punto.confort_score > 40 ? 6 : 4;

        let marker = L.circleMarker([punto.latitude, punto.longitude], {
          radius: radius,
          fillColor: color,
          fillOpacity: 0.7,
          color: '#000',
          weight: 1
        })
          .bindPopup(`
            <strong>${punto.Location}</strong><br>
            Confort: ${punto.etiqueta_confort}<br>
            Score: ${punto.confort_score}/100<br>
            Temp: ${punto.MaxTemp}°C<br>
            Humedad: ${punto.Humidity3pm}%
          `)
          .addTo(mapa);

        markers.push(marker);
      }
    });

    // Mostrar descripción de resultados
    document.getElementById('resultDescription').innerHTML = `
      <i class="fas fa-home"></i> Este análisis identifica qué zonas de Australia no son confortables para vivir basándose en condiciones climáticas.
      <br><br>
      Se analizaron variables como <strong>temperatura máxima</strong>, <strong>humedad</strong>, <strong>velocidad del viento</strong>, 
      <strong>horas de sol</strong> y <strong>lluvia</strong> para determinar el nivel de confort de cada zona.
      <br><br>
      El <strong>Score de Confort</strong> va de 0 a 100, donde valores más altos indican mayor confort para vivir.
    `;

    // Crear gráfico de confort
    const ctx = document.getElementById('grafico').getContext('2d');
    const confortLabels = Object.keys(confort_stats);
    const confortCounts = Object.values(confort_stats);
    const colores = confortLabels.map(etiqueta => 
      etiqueta === 'CONFORME' ? '#27ae60' : '#e74c3c'
    );

    chart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: confortLabels,
        datasets: [{
          data: confortCounts,
          backgroundColor: colores,
          borderWidth: 2,
          borderColor: '#fff'
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { 
            display: true,
            position: 'bottom'
          },
          title: { 
            display: true, 
            text: 'Distribución de Confort por Zona' 
          }
        }
      }
    });
  }

      function mostrarResultadosLluvia(data) {
        // Contar predicciones
        let predicciones = { 'Sí lloverá': 0, 'No lloverá': 0 };
        let probabilidades = [];

        data.forEach(punto => {
          if (punto.llovera_manana !== undefined) {
            if (punto.llovera_manana) {
              predicciones['Sí lloverá']++;
            } else {
              predicciones['No lloverá']++;
            }
            if (punto.probabilidad_lluvia !== undefined) {
              probabilidades.push(punto.probabilidad_lluvia);
            }
          }
        });

        // Mostrar información de predicciones
        document.getElementById('clima-info').innerHTML = `
          <div class="clima-box">
            <div><span class="clima-color" style="background: #3498db;"></span><strong>Sí lloverá mañana</strong></div>
            <p>${predicciones['Sí lloverá']} localidades</p>
          </div>
          <div class="clima-box">
            <div><span class="clima-color" style="background: #e74c3c;"></span><strong>No lloverá mañana</strong></div>
            <p>${predicciones['No lloverá']} localidades</p>
          </div>
        `;

        // Mostrar marcadores en el mapa
        data.forEach(punto => {
          if (punto.latitude !== undefined && punto.longitude !== undefined) {
            let color = punto.llovera_manana ? '#3498db' : '#e74c3c';
            let icon = punto.llovera_manana ? '☔' : '☀️';
            
            let marker = L.circleMarker([punto.latitude, punto.longitude], {
              radius: 6,
              fillColor: color,
              fillOpacity: 0.7,
              color: '#000',
              weight: 1
            })
              .bindPopup(`
                <strong>${punto.Location}</strong><br>
                ${icon} ${punto.llovera_manana ? 'Sí lloverá' : 'No lloverá'}<br>
                Probabilidad: ${(punto.probabilidad_lluvia * 100).toFixed(1)}%<br>
                MaxTemp: ${punto.MaxTemp}°C<br>
                Humedad: ${punto.Humidity3pm}%
              `)
              .addTo(mapa);

            markers.push(marker);
          }
        });

        // Mostrar descripción de resultados
        document.getElementById('resultDescription').innerHTML = `
          <i class="fas fa-cloud-rain"></i> Este análisis predice si lloverá mañana basándose en las condiciones climáticas actuales.
          <br><br>
          Se utilizó un modelo de <strong>Regresión Logística</strong> entrenado con variables como temperatura máxima, humedad, 
          presión atmosférica y velocidad del viento para realizar las predicciones.
          <br><br>
          Los marcadores azules indican localidades donde se predice lluvia, mientras que los rojos indican días sin lluvia.
          La probabilidad mostrada representa la confianza del modelo en cada predicción.
        `;

        // Crear gráfico de predicciones
        const ctx = document.getElementById('grafico').getContext('2d');
        chart = new Chart(ctx, {
          type: 'doughnut',
          data: {
            labels: ['Sí lloverá', 'No lloverá'],
            datasets: [{
              data: [predicciones['Sí lloverá'], predicciones['No lloverá']],
              backgroundColor: ['#3498db', '#e74c3c']
            }]
          },
          options: {
            responsive: true,
            plugins: {
              legend: {
                position: 'bottom'
              },
              title: { 
                display: true, 
                text: 'Distribución de Predicciones de Lluvia' 
              }
            }
          }
        });
      }

      function mostrarResultadosEvaporacion(data) {
        // Contar niveles de evaporación
        let niveles = { 'Baja': 0, 'Media': 0, 'Alta': 0 };
        let evaporaciones = [];

        data.forEach(punto => {
          if (punto.nivel_evaporacion) {
            niveles[punto.nivel_evaporacion]++;
          }
          if (punto.evaporacion_predicha !== undefined) {
            evaporaciones.push(punto.evaporacion_predicha);
          }
        });

        // Mostrar información de niveles
        document.getElementById('clima-info').innerHTML = `
          <div class="clima-box">
            <div><span class="clima-color" style="background: #2ecc71;"></span><strong>Evaporación Baja</strong></div>
            <p>${niveles['Baja']} localidades</p>
          </div>
          <div class="clima-box">
            <div><span class="clima-color" style="background: #f39c12;"></span><strong>Evaporación Media</strong></div>
            <p>${niveles['Media']} localidades</p>
          </div>
          <div class="clima-box">
            <div><span class="clima-color" style="background: #e74c3c;"></span><strong>Evaporación Alta</strong></div>
            <p>${niveles['Alta']} localidades</p>
          </div>
        `;

        // Mostrar marcadores en el mapa
        data.forEach(punto => {
          if (punto.latitude !== undefined && punto.longitude !== undefined) {
            let color;
            let icon;
            switch(punto.nivel_evaporacion) {
              case 'Baja':
                color = '#2ecc71';
                icon = '💧';
                break;
              case 'Media':
                color = '#f39c12';
                icon = '🌡️';
                break;
              case 'Alta':
                color = '#e74c3c';
                icon = '🔥';
                break;
              default:
                color = '#95a5a6';
                icon = '❓';
            }
            
            let marker = L.circleMarker([punto.latitude, punto.longitude], {
              radius: 6,
              fillColor: color,
              fillOpacity: 0.7,
              color: '#000',
              weight: 1
            })
              .bindPopup(`
                <strong>${punto.Location}</strong><br>
                ${icon} Nivel: ${punto.nivel_evaporacion}<br>
                Evaporación predicha: ${punto.evaporacion_predicha.toFixed(2)} mm<br>
                MaxTemp: ${punto.MaxTemp}°C<br>
                Humedad: ${punto.Humidity3pm}%<br>
                Sol: ${punto.Sunshine} horas
              `)
              .addTo(mapa);

            markers.push(marker);
          }
        });

        // Mostrar descripción de resultados
        document.getElementById('resultDescription').innerHTML = `
          <i class="fas fa-thermometer-half"></i> Este análisis predice los niveles de evaporación basándose en variables climáticas como temperatura, humedad y horas de sol.
          <br><br>
          Se utilizó un modelo de <strong>Regresión</strong> entrenado con variables como temperatura máxima, humedad, 
          velocidad del viento y horas de sol para predecir la evaporación diaria.
          <br><br>
          Los marcadores verdes indican evaporación baja, los amarillos evaporación media, y los rojos evaporación alta.
          La evaporación predicha se mide en milímetros por día.
        `;

        // Crear gráfico de niveles
        const ctx = document.getElementById('grafico').getContext('2d');
        chart = new Chart(ctx, {
          type: 'bar',
          data: {
            labels: ['Baja', 'Media', 'Alta'],
            datasets: [{
              label: 'Número de localidades',
              data: [niveles['Baja'], niveles['Media'], niveles['Alta']],
              backgroundColor: ['#2ecc71', '#f39c12', '#e74c3c']
            }]
          },
          options: {
            responsive: true,
            plugins: {
              legend: { display: false },
              title: { display: true, text: 'Distribución de Niveles de Evaporación' }
            },
            scales: {
              y: { beginAtZero: true, precision: 0 }
            }
          }
        });
      }

        function mostrarResultadosAsociacion(data) {
    // Contar patrones únicos
    let patrones = {};
    data.forEach(punto => {
      if (punto.patron_climatico) {
        if (patrones[punto.patron_climatico]) {
          patrones[punto.patron_climatico]++;
        } else {
          patrones[punto.patron_climatico] = 1;
        }
      }
    });

    // Mostrar información de patrones
    let patronesHTML = '';
    Object.keys(patrones).forEach(patron => {
      const color = patron.includes('Patrón 1') ? '#e74c3c' : 
                   patron.includes('Patrón 2') ? '#3498db' : 
                   patron.includes('Patrón 3') ? '#2ecc71' : '#95a5a6';
      
      patronesHTML += `
        <div class="clima-box">
          <div><span class="clima-color" style="background: ${color};"></span><strong>${patron}</strong></div>
          <p>Frecuencia: ${patrones[patron]} localidades</p>
        </div>
      `;
    });

    document.getElementById('clima-info').innerHTML = patronesHTML;

    // Mostrar marcadores en el mapa
    data.forEach(punto => {
      if (punto.latitude !== undefined && punto.longitude !== undefined) {
        let color = '#95a5a6'; // Color por defecto
        if (punto.patron_climatico.includes('Patrón 1')) color = '#e74c3c';
        else if (punto.patron_climatico.includes('Patrón 2')) color = '#3498db';
        else if (punto.patron_climatico.includes('Patrón 3')) color = '#2ecc71';

        let marker = L.circleMarker([punto.latitude, punto.longitude], {
          radius: 6,
          fillColor: color,
          fillOpacity: 0.7,
          color: '#000',
          weight: 1
        })
          .bindPopup(`<strong>${punto.Location}</strong><br>Patrón: ${punto.patron_climatico}`)
          .addTo(mapa);

        markers.push(marker);
      }
    });

    // Mostrar descripción de resultados
    document.getElementById('resultDescription').innerHTML = `
      <i class="fas fa-link"></i> Este análisis identifica las combinaciones de condiciones climáticas que suelen ocurrir juntas en un mismo día.
      <br><br>
      Se analizaron las variables: <strong>MaxTemp</strong>, <strong>Humidity3pm</strong>, <strong>WindGustSpeed</strong> y <strong>Sunshine</strong>, 
      categorizándolas en niveles Baja, Media y Alta para encontrar patrones frecuentes.
      <br><br>
      Los patrones identificados representan las combinaciones más comunes de estas variables climáticas en los datos analizados.
    `;

    // Crear gráfico de patrones
    const ctx = document.getElementById('grafico').getContext('2d');
    const patronesLabels = Object.keys(patrones);
    const patronesCounts = Object.values(patrones);
    const colores = patronesLabels.map(patron => 
      patron.includes('Patrón 1') ? '#e74c3c' : 
      patron.includes('Patrón 2') ? '#3498db' : 
      patron.includes('Patrón 3') ? '#2ecc71' : '#95a5a6'
    );

    chart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: patronesLabels,
        datasets: [{
          label: 'Número de localidades',
          data: patronesCounts,
          backgroundColor: colores
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          title: { display: true, text: 'Distribución de Patrones Climáticos' }
        },
        scales: {
          y: { beginAtZero: true, precision: 0 }
        }
      }
    });
  }

      // Función para verificar si el botón debe estar habilitado
      function verificarEstadoBoton() {
        const fileInput = document.getElementById('fileInput');
        const btnEnviar = document.getElementById('btnEnviar');
        const file = fileInput.files[0];
        
        // El botón se habilita solo si hay técnica seleccionada Y archivo seleccionado
        if (file && currentTechnique) {
          btnEnviar.disabled = false;
          btnEnviar.innerHTML = '<i class="fas fa-play"></i> Iniciar Análisis';
        } else {
          btnEnviar.disabled = true;
          btnEnviar.innerHTML = '<i class="fas fa-play"></i> Iniciar Análisis';
        }
      }

      // Manejar selección de archivo
      document.getElementById('fileInput').addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
          console.log('Archivo seleccionado:', file.name);
        }
        verificarEstadoBoton();
      });

      // Manejar envío de archivo
      document.getElementById('btnEnviar').addEventListener('click', function() {
        const fileInput = document.getElementById('fileInput');
        const file = fileInput.files[0];
        
        if (!file) {
          alert('No se seleccionó ningún archivo.');
          return;
        }

        if (!currentTechnique) {
          alert('Por favor selecciona una técnica de análisis primero.');
          return;
        }

        Papa.parse(file, {
          header: true,
          dynamicTyping: true,
          complete: function (results) {
            const datos = results.data;

            // Filtrar datos válidos
            const datosFiltrados = datos.filter(fila => {
              return Object.values(fila).some(val => val !== null && val !== undefined && val !== '');
            });

            if (datosFiltrados.length === 0) {
              alert('El archivo no contiene datos válidos.');
              return;
            }

            // Enviar datos al servidor
            fetch('/data', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                technique: currentTechnique,
                data: datosFiltrados
              })
            })
            .then(async response => {
              const text = await response.text();
              if (!response.ok) {
                try {
                  const errorData = JSON.parse(text);
                  throw new Error(errorData.error || 'Error en el servidor');
                } catch {
                  throw new Error('Error inesperado del servidor');
                }
              }

              let data;
              try {
                data = JSON.parse(text);
              } catch {
                throw new Error('Respuesta no es JSON válido');
              }

              if (!Array.isArray(data)) {
                alert('Formato de datos inesperado.');
                return;
              }

              mostrarDatos(data);
            })
            .catch(err => {
              console.error('Error al procesar los datos:', err);
              alert('Error: ' + err.message);
            });
          }
        });
      });
    });
