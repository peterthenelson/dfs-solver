
pub fn color_ave2(a: (u8, u8, u8), b: (u8, u8, u8)) -> (u8, u8, u8) {
    let r = ((a.0 as f32)*0.5 + (b.0 as f32)*0.5) as u8;
    let g = ((a.1 as f32)*0.5 + (b.1 as f32)*0.5) as u8;
    let b = ((a.2 as f32)*0.5 + (b.2 as f32)*0.5) as u8;
    (r, g, b)
}

pub fn color_ave(colors: &Vec<(u8, u8, u8)>) -> (u8, u8, u8) {
    let sum = colors.iter().fold((0.0, 0.0, 0.0), |a, b| {
        (a.0 + b.0 as f32, a.1 + b.1 as f32, a.2 + b.2 as f32)
    });
    let n = colors.len() as f32;
    ((sum.0/n) as u8, (sum.1/n) as u8, (sum.2/n) as u8)
}

pub fn color_lerp(a: (u8, u8, u8), b: (u8, u8, u8), ratio: f32) -> (u8, u8, u8) {
    let (t, tc) = (ratio, 1.0 - ratio);
    let r = ((a.0 as f32)*t + (b.0 as f32)*tc) as u8;
    let g = ((a.1 as f32)*t + (b.1 as f32)*tc) as u8;
    let b = ((a.2 as f32)*t + (b.2 as f32)*tc) as u8;
    (r, g, b)
}

// Green is low, orange is high, pink is exact middle, if any.
pub fn polarity_color(min: u8, max: u8, v: u8) -> (u8, u8, u8) {
    let mid = ((min as f32) + (max as f32))/2.0;
    if (v as f32) < mid {
        (0, 200, 0)
    } else if (v as f32) > mid {
        (200, 130, 0)
    } else {
        (200, 0, 200)
    }
}