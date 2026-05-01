use plotters::prelude::*;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let input = std::env::args().nth(1).unwrap_or_else(|| "loss.csv".to_string());
    let output = std::env::args().nth(2).unwrap_or_else(|| "loss.svg".to_string());

    let data = load_loss(&input)?;
    if data.is_empty() {
        eprintln!("No finite loss values found in {input}");
        std::process::exit(1);
    }

    plot_loss(&data, &output)?;
    println!("Saved → {output}");
    Ok(())
}

fn load_loss(path: &str) -> Result<Vec<(f64, f64)>, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut out = Vec::new();
    for result in rdr.records() {
        let rec = result?;
        let epoch: f64 = rec[0].trim().parse()?;
        let pct: f64 = rec[1].trim().parse()?;
        let loss: f64 = rec[2].trim().parse()?;
        if loss.is_finite() {
            // Map to a continuous x so intra-epoch progress is visible
            out.push((epoch - 1.0 + pct / 100.0, loss));
        }
    }
    Ok(out)
}

fn plot_loss(data: &[(f64, f64)], output: &str) -> Result<(), Box<dyn Error>> {
    let x_max = data.iter().copied().map(|(x, _)| x).fold(f64::NEG_INFINITY, f64::max);
    let y_min = data.iter().copied().map(|(_, y)| y).fold(f64::INFINITY, f64::min);
    let y_max = data.iter().copied().map(|(_, y)| y).fold(f64::NEG_INFINITY, f64::max);
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
        .x_desc("Epoch")
        .y_desc("Loss (MSE)")
        .draw()?;

    chart
        .draw_series(LineSeries::new(data.iter().copied(), &BLUE))?
        .label("train loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}
