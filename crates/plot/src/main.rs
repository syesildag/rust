use plotters::prelude::*;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let input = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "loss.csv".to_string());
    let output = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "loss.svg".to_string());
    let window: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);

    let data = load_loss(&input)?;
    if data.is_empty() {
        eprintln!("No finite loss values found in {input}");
        std::process::exit(1);
    }

    plot_loss(&data, &output, window)?;
    println!("Saved → {output}");
    Ok(())
}

fn load_loss(path: &str) -> Result<Vec<(f64, f64)>, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut rows: Vec<(f64, f64)> = Vec::new();
    for result in rdr.records() {
        let rec = result?;
        let pct: f64 = rec[1].trim().parse()?;
        let loss: f64 = rec[2].trim().parse()?;
        if loss.is_finite() {
            rows.push((pct, loss));
        }
    }
    let n = rows.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let max_pct = rows.iter().map(|&(p, _)| p).fold(f64::NEG_INFINITY, f64::max);
    Ok(rows
        .into_iter()
        .enumerate()
        .map(|(i, (_, loss))| {
            #[allow(clippy::cast_precision_loss)]
            let x = i as f64 / (n - 1) as f64 * max_pct;
            (x, loss)
        })
        .collect())
}

// Trailing moving average: each output point is the mean of the previous `window` raw points.
// Returns a slice-aligned vec (same length) — early points use whatever is available.
fn moving_average(data: &[(f64, f64)], window: usize) -> Vec<(f64, f64)> {
    data.iter()
        .enumerate()
        .map(|(i, &(x, _))| {
            let start = i.saturating_sub(window - 1);
            let slice = &data[start..=i];
            #[allow(clippy::cast_precision_loss)]
            let mean = slice.iter().map(|&(_, y)| y).sum::<f64>() / slice.len() as f64;
            (x, mean)
        })
        .collect()
}

fn plot_loss(data: &[(f64, f64)], output: &str, window: usize) -> Result<(), Box<dyn Error>> {
    let x_max = data
        .iter()
        .copied()
        .map(|(x, _)| x)
        .fold(f64::NEG_INFINITY, f64::max);
    let y_min = data
        .iter()
        .copied()
        .map(|(_, y)| y)
        .fold(f64::INFINITY, f64::min);
    let y_max = data
        .iter()
        .copied()
        .map(|(_, y)| y)
        .fold(f64::NEG_INFINITY, f64::max);
    let y_pad = (y_max - y_min).max(1e-6) * 0.1;

    let root = SVGBackend::new(output, (1200, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Training Loss", ("sans-serif", 36).into_font())
        .margin(30)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(0.0..x_max, (y_min - y_pad)..(y_max + y_pad))?;

    chart
        .configure_mesh()
        .x_desc("Epoch Progress (%)")
        .y_desc("Loss (MSE)")
        .draw()?;

    if window > 1 {
        // Draw raw data as a faint background series
        chart.draw_series(LineSeries::new(
            data.iter().copied(),
            RGBAColor(180, 200, 230, 0.25),
        ))?;

        let smoothed = moving_average(data, window);
        chart
            .draw_series(LineSeries::new(smoothed.iter().copied(), &BLUE))?
            .label(format!("MA-{window}"))
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));
    } else {
        chart
            .draw_series(LineSeries::new(data.iter().copied(), &BLUE))?
            .label("train loss")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));
    }

    chart.configure_series_labels().border_style(BLACK).draw()?;

    root.present()?;
    Ok(())
}
