%% dynamic_constellation_cloud_lighting_simulation.m
% Dynamic POI / small-satellite constellation simulation with:
%   - static fixed-inclination circular LEO orbits
%   - dynamic POI arrivals
%   - agricultural-region POI placement
%   - stochastic regional cloud obstruction
%   - solar-lighting constraints using solar zenith angle
%   - NDVI first-pass check
%   - 6-band revisit requirement for suspicious POIs
%   - constellation-size comparison using "resolved in < 2 days"
%
%
% Simplifications:
%   1) Ccircular LEO satellites.
%   2) Inclination is fixed once chosen.
%   3) Imaging is allowed when the satellite subpoint is within a radius
%      around a POI, the POI is sufficiently sunlit, and the POI is cloud-free.
%   4) Cloud cover is stochastic and region-dependent (not pulled from real
%      weather data)
%   5) OPERA/HLS values are represented by simulated NDVI values for now.
%   6) No slew limits, power limits, storage limits, downlink limits, or
%      tasking conflicts are included yet.

clear; clc; close all;

%% --------------------------- USER SETTINGS ---------------------------- %%
rng(11);                         % Reproducible POIs, clouds, NDVI values

% POI arrival window and simulation window
poiArrivalDays = 60;             % New POIs are created during this many days
fastResolveLimit_days = 4;       % Metric: resolved in less than this many days
finalFollowupDays = fastResolveLimit_days;
totalSimDays = poiArrivalDays + finalFollowupDays;

sampleTime_s = 120;              % 2-minute timestep

% Dynamic POI process
initialPOIs = 15;
newPOIsPerDay = 7;
clusterPOIProbability = 0.70;    % Probability that a day's new POIs are near each other
clusterSpread_deg = 1.2;         % Approx geographic spread for clustered POIs

% Imaging geometry
poiAccessRadius_km = 175;        % Satellite subpoint must be within this radius
                                 % of a POI to image it.

% Lighting constraint
maxSolarZenith_deg = 70;         % Imaging allowed only if solar zenith <= this.
                                 % 70 deg zenith = 20 deg sun elevation.

startDatetimeUTC = datetime(2026,6,1,0,0,0);  % Approx UTC start date.
                                               % Change this to test seasons.

% Cloud model
cloudinessScale = 1.00;          % >1 cloudier, <1 clearer
cloudDailySigma = 0.08;          % Day-to-day random cloud-probability variation
cloudPersistence_hr = 6;         % Larger value = clouds persist longer in time
cloudProbabilityFloor = 0.03;
cloudProbabilityCeil = 0.95;

% NDVI decision rule
ndviLowThreshold = 0.35;         % If measured NDVI < this, something is wrong
hlsEpsilon = 0.15;               % If |measured NDVI - HLS NDVI| > epsilon, flag it
minSixBandRevisitDelay_hr = 1;   % Prevents "NDVI check" and "6-band revisit"
                                 % from resolving in the exact same pass.

% Constellation sizing
numSatsGrid = 1:12;              % Try these constellation sizes
goalFastResolutionFraction = 0.80; % Placeholder "good enough" fast-resolution target

% Candidate constellation architecture:
%   'singlePlanePhased' = same RAAN, satellites phased along same orbit
%   'balancedPlanes'   = satellites split across RAAN-spaced planes
architectureModes = {'singlePlanePhased', 'balancedPlanes'};

% Fixed circular LEO orbit
altitude_km = 550;
inclination_deg = 40;            % Crop-focused. Try 40, 50, 60, or 98.
useJ2Precession = true;

% Earth / orbital constants
Re_km = 6378.137;
mu_km3_s2 = 398600.4418;
we_rad_s = 7.2921159e-5;
J2 = 1.08262668e-3;

%% ------------------------ PRE-GENERATE POIS -------------------------- %%

poiCatalog = generatePOICatalog( ...
    initialPOIs, newPOIsPerDay, poiArrivalDays, ...
    clusterPOIProbability, clusterSpread_deg);

numPOI_total = height(poiCatalog);

% Simulated HLS values cause I'm too lazy to get actual OPERA/HLS-driven information rn
poiCatalog = assignNDVIValues(poiCatalog, ndviLowThreshold, hlsEpsilon);

%% ----------------------------- SIM TIME ------------------------------ %%
t_s = (0:sampleTime_s:totalSimDays*86400).';
numTime = numel(t_s);

%% ------------------- PRECOMPUTE CLOUDS AND LIGHTING ----------------- %%
% These are generated once so every constellation-size case faces the same cloud/lighting conditions.

lightingOK = generateLightingOK( ...
    poiCatalog, t_s, startDatetimeUTC, maxSolarZenith_deg);

cloudBlocked = generateCloudBlockage( ...
    poiCatalog, t_s, totalSimDays, cloudinessScale, cloudDailySigma, ...
    cloudPersistence_hr, cloudProbabilityFloor, cloudProbabilityCeil);

clearAndLit = lightingOK & ~cloudBlocked;
meanClearLitFraction = mean(clearAndLit(:));

fprintf('\n========= DYNAMIC CONSTELLATION / CLOUD + LIGHTING SIMULATION =========\n');
fprintf('POI arrival window:            %d days\n', poiArrivalDays);
fprintf('Final follow-up window:        %.1f days\n', finalFollowupDays);
fprintf('Total simulated time:          %.1f days\n', totalSimDays);
fprintf('Total POIs arriving:           %d\n', numPOI_total);
fprintf('Initial POIs:                  %d\n', initialPOIs);
fprintf('New POIs per day:              %d\n', newPOIsPerDay);
fprintf('POI access radius:             %.0f km\n', poiAccessRadius_km);
fprintf('Max solar zenith angle:        %.0f deg\n', maxSolarZenith_deg);
fprintf('Cloudiness scale:              %.2f\n', cloudinessScale);
fprintf('Cloud persistence:             %.1f hr\n', cloudPersistence_hr);
fprintf('Mean clear-and-lit fraction:   %.3f\n', meanClearLitFraction);
fprintf('Fast-resolution threshold:     < %.1f days\n', fastResolveLimit_days);
fprintf('Orbit altitude:                %.0f km\n', altitude_km);
fprintf('Fixed inclination:             %.0f deg\n', inclination_deg);
fprintf('J2 RAAN precession:            %d\n\n', useJ2Precession);

%% ---------------------- RUN CONSTELLATION STUDY ---------------------- %%
resultRows = {};

allScenarioOutputs = struct();
scenarioCounter = 0;

for m = 1:numel(architectureModes)
    architectureMode = architectureModes{m};

    for Nsat = numSatsGrid
        scenarioCounter = scenarioCounter + 1;

        satDef = makeConstellation(Nsat, altitude_km, inclination_deg, architectureMode);

        [latSS_deg, lonSS_deg] = propagateConstellationECEF( ...
            satDef, t_s, Re_km, mu_km3_s2, we_rad_s, J2, useJ2Precession);

        simOut = runPOISimulation( ...
            poiCatalog, latSS_deg, lonSS_deg, t_s, Re_km, poiAccessRadius_km, ...
            lightingOK, cloudBlocked, ndviLowThreshold, hlsEpsilon, ...
            fastResolveLimit_days, minSixBandRevisitDelay_hr);

        resultRows(end+1,:) = { ...
            architectureMode, Nsat, satDef.numPlanes, simOut.totalArrived, ...
            simOut.numResolved, simOut.resolutionFraction, ...
            simOut.numResolvedFast, simOut.fastResolutionFraction, ...
            simOut.numResolvedImmediate, simOut.numResolvedAfterSixBand, ...
            simOut.numUnresolvedEnd, simOut.meanResolutionTime_days, ...
            simOut.medianResolutionTime_days, simOut.meanBacklog, ...
            simOut.numMultiPOIImagingEvents};

        allScenarioOutputs(scenarioCounter).architectureMode = architectureMode;
        allScenarioOutputs(scenarioCounter).Nsat = Nsat;
        allScenarioOutputs(scenarioCounter).satDef = satDef;
        allScenarioOutputs(scenarioCounter).latSS_deg = latSS_deg;
        allScenarioOutputs(scenarioCounter).lonSS_deg = lonSS_deg;
        allScenarioOutputs(scenarioCounter).simOut = simOut;
    end
end

resultsTable = cell2table(resultRows, ...
    'VariableNames', {'Architecture','NumSats','NumPlanes','TotalArrived', ...
    'NumResolved','ResolutionFraction','NumResolvedFast','FastResolutionFraction', ...
    'ResolvedImmediate','ResolvedAfterSixBand','UnresolvedEnd', ...
    'MeanResolutionTime_days','MedianResolutionTime_days','MeanBacklog', ...
    'MultiPOIImagingEvents'});

resultsTable = sortrows(resultsTable, {'Architecture','NumSats'});

disp('Constellation sizing results:');
disp(resultsTable);

%% -------------------- RECOMMENDED CONSTELLATION SIZE ----------------- %%
% Pick the smallest constellation that reaches the placeholder target.

meetsGoal = resultsTable.FastResolutionFraction >= goalFastResolutionFraction;

if any(meetsGoal)
    candidates = resultsTable(meetsGoal,:);
    candidates = sortrows(candidates, ...
        {'NumSats','MeanResolutionTime_days'}, {'ascend','ascend'});
    recommended = candidates(1,:);
    fprintf('\nSmallest case meeting %.0f%% resolved in < %.1f days:\n', ...
        100*goalFastResolutionFraction, fastResolveLimit_days);
else
    candidates = sortrows(resultsTable, ...
        {'FastResolutionFraction','MeanResolutionTime_days'}, {'descend','ascend'});
    recommended = candidates(1,:);
    fprintf('\nNo case met %.0f%% resolved in < %.1f days. Best available case:\n', ...
        100*goalFastResolutionFraction, fastResolveLimit_days);
end

disp(recommended);

% Find matching saved scenario for plots.
plotScenarioIdx = findScenario(allScenarioOutputs, recommended.Architecture{1}, recommended.NumSats);
plotScenario = allScenarioOutputs(plotScenarioIdx);

%% ------------------------------- PLOTS ------------------------------- %%
% 1) Fast resolution fraction versus number of satellites
figure('Name','Fast Resolution Fraction vs Number of Satellites');
hold on;
for m = 1:numel(architectureModes)
    mode = architectureModes{m};
    idx = strcmp(resultsTable.Architecture, mode);
    plot(resultsTable.NumSats(idx), resultsTable.FastResolutionFraction(idx), ...
        '-o', 'LineWidth', 1.5);
end
plot([min(numSatsGrid), max(numSatsGrid)], ...
     [goalFastResolutionFraction, goalFastResolutionFraction], ...
     'k--', 'LineWidth', 1.1);

grid on;
xlabel('Number of satellites');
ylabel(sprintf('Fraction of POIs resolved in < %.1f days', fastResolveLimit_days));
ylim([0 1]);
legend([architectureModes, {'Goal'}], 'Location', 'southeast');
title(sprintf('Fast POI resolution vs constellation size, with clouds and lighting'));

% 2) Mean resolution time versus number of satellites
figure('Name','Mean Resolution Time vs Number of Satellites');
hold on;
for m = 1:numel(architectureModes)
    mode = architectureModes{m};
    idx = strcmp(resultsTable.Architecture, mode);
    plot(resultsTable.NumSats(idx), resultsTable.MeanResolutionTime_days(idx), ...
        '-o', 'LineWidth', 1.5);
end
grid on;
xlabel('Number of satellites');
ylabel('Mean resolution time [days]');
legend(architectureModes, 'Location', 'northeast');
title('Mean time from POI arrival to resolution');

% 3) Backlog over time for recommended case
figure('Name','Unresolved POI Backlog');
plot(plotScenario.simOut.backlogTime_days, plotScenario.simOut.backlogCount, ...
    'LineWidth', 1.5);
grid on;
xlabel('Simulation time [days]');
ylabel('Unresolved active POIs');
title(sprintf('Backlog over time | %s | %d satellites', ...
    plotScenario.architectureMode, plotScenario.Nsat));

% 4) First-day ground tracks and first-day POI access circles
figure('Name','Recommended Constellation: First-Day Ground Tracks');
hold on;

firstDayIdx = t_s <= 86400;
Nplot = plotScenario.Nsat;

for s = 1:Nplot
    [lonPlot, latPlot] = breakTrackAtDateLine( ...
        plotScenario.lonSS_deg(firstDayIdx,s), plotScenario.latSS_deg(firstDayIdx,s));
    plot(lonPlot, latPlot, 'LineWidth', 1.1);
end

firstDayPOIs = poiCatalog.ArrivalTime_s <= 86400;
scatter(poiCatalog.Longitude_deg(firstDayPOIs), poiCatalog.Latitude_deg(firstDayPOIs), ...
    70, 'k', 'filled');

% Plot access-radius circles around first-day POIs.
for k = find(firstDayPOIs).'
    [circleLat, circleLon] = smallCircleLatLon( ...
        poiCatalog.Latitude_deg(k), poiCatalog.Longitude_deg(k), poiAccessRadius_km, Re_km);
    plot(circleLon, circleLat, 'k:', 'LineWidth', 0.9);
end

grid on;
box on;
xlim([-180 180]);
ylim([-60 60]);
xlabel('Longitude [deg]');
ylabel('Latitude [deg]');
title(sprintf(['First-day ground tracks and POI access circles | %s | ', ...
    '%d sats, %.0f planes, i = %.0f deg'], ...
    plotScenario.architectureMode, plotScenario.Nsat, ...
    plotScenario.satDef.numPlanes, inclination_deg));

% 5) Final POI status map for recommended case
figure('Name','Final POI Status Map');
hold on;

resolved = plotScenario.simOut.finalStatus == 3;
needsNDVI = plotScenario.simOut.finalStatus == 1;
needs6 = plotScenario.simOut.finalStatus == 2;

resolvedFast = resolved & ...
    ((plotScenario.simOut.resolvedTime_s - poiCatalog.ArrivalTime_s) < ...
    fastResolveLimit_days*86400);

scatter(poiCatalog.Longitude_deg(resolvedFast), poiCatalog.Latitude_deg(resolvedFast), ...
    65, 'g', 'filled');
scatter(poiCatalog.Longitude_deg(resolved & ~resolvedFast), poiCatalog.Latitude_deg(resolved & ~resolvedFast), ...
    65, 'b', 'filled');
scatter(poiCatalog.Longitude_deg(needsNDVI), poiCatalog.Latitude_deg(needsNDVI), ...
    65, 'r', 'filled');
scatter(poiCatalog.Longitude_deg(needs6), poiCatalog.Latitude_deg(needs6), ...
    65, 'm', 'filled');

grid on;
box on;
xlim([-180 180]);
ylim([-60 60]);
xlabel('Longitude [deg]');
ylabel('Latitude [deg]');
legend({'Resolved < 2 days','Resolved, but slower','Still needs NDVI','Still needs 6-band revisit'}, ...
    'Location', 'bestoutside');
title(sprintf('Final POI status | %s | %d satellites', ...
    plotScenario.architectureMode, plotScenario.Nsat));

% 6) Example cloud/lighting availability for POIs
figure('Name','POI Clear-and-Lit Availability');
availabilityByPOI = mean(clearAndLit, 2);
histogram(availabilityByPOI, 12);
grid on;
xlabel('Fraction of time POI is both clear and sufficiently sunlit');
ylabel('Number of POIs');
title('Cloud + lighting availability across simulated POIs');

%% --------------------------- LOCAL FUNCTIONS ------------------------- %%
function poiCatalog = generatePOICatalog( ...
    initialPOIs, newPOIsPerDay, poiArrivalDays, clusterPOIProbability, clusterSpread_deg)

    totalPOIs = initialPOIs + newPOIsPerDay*(poiArrivalDays - 1);

    lat = zeros(totalPOIs,1);
    lon = zeros(totalPOIs,1);
    arrivalTime_s = zeros(totalPOIs,1);
    arrivalDay = zeros(totalPOIs,1);
    regionID = zeros(totalPOIs,1);
    regionName = strings(totalPOIs,1);
    baseCloudProbability = zeros(totalPOIs,1);

    idx = 0;

    % Day 1: initial POIs
    for k = 1:initialPOIs
        idx = idx + 1;
        [lat(idx), lon(idx), regionID(idx), regionName(idx), baseCloudProbability(idx)] = ...
            sampleAgriculturalLocation();
        arrivalDay(idx) = 1;
        arrivalTime_s(idx) = 0;
    end

    % Subsequent days: add new POIs
    for d = 2:poiArrivalDays
        useCluster = rand() < clusterPOIProbability;

        if useCluster
            parentRegionID = sampleRegionID();
            [centerLat, centerLon, ~, ~, ~] = sampleAgriculturalLocation(parentRegionID);
        end

        for j = 1:newPOIsPerDay
            idx = idx + 1;

            if useCluster
                [lat(idx), lon(idx), regionID(idx), regionName(idx), baseCloudProbability(idx)] = ...
                    sampleAgriculturalLocationNear(parentRegionID, centerLat, centerLon, clusterSpread_deg);
            else
                [lat(idx), lon(idx), regionID(idx), regionName(idx), baseCloudProbability(idx)] = ...
                    sampleAgriculturalLocation();
            end

            arrivalDay(idx) = d;
            arrivalTime_s(idx) = (d-1)*86400;
        end
    end

    poiID = (1:totalPOIs).';
    poiCatalog = table(poiID, arrivalDay, arrivalTime_s, lat, lon, ...
        regionID, regionName, baseCloudProbability, ...
        'VariableNames', {'POI_ID','ArrivalDay','ArrivalTime_s', ...
        'Latitude_deg','Longitude_deg','RegionID','RegionName','BaseCloudProbability'});
end

function regions = agriculturalRegions()
    % Columns are broad agricultural regions
    % Longitudes are degrees east; west is negative.
    % BaseCloudProbability is a rough stochastic model parameter

    Name = [
        "California Central Valley"
        "US Midwest Corn Belt"
        "US Great Plains"
        "Mississippi Delta"
        "Mexico Bajio"
        "Colombia Ecuador Valleys"
        "Brazil Cerrado"
        "Southern Brazil Agriculture"
        "Argentina Pampas"
        "Chile Central Valley"
        "France Spain Agriculture"
        "Italy Po Valley"
        "Ukraine Black Sea Agriculture"
        "Turkey Anatolia"
        "Nile Valley Delta"
        "Morocco Algeria Agriculture"
        "West Africa Cropland Belt"
        "Ethiopia Highlands"
        "East Africa Cropland Belt"
        "South Africa Highveld"
        "Indo Gangetic West"
        "Indo Gangetic East"
        "Pakistan Punjab Sindh"
        "India Deccan Plateau"
        "Bangladesh Northeast India"
        "Mekong Delta Vietnam"
        "Thailand Cambodia Rice"
        "Java Indonesia"
        "North China Plain"
        "Sichuan Basin"
        "Northeast China Agriculture"
        "Korean Peninsula Cropland"
        "Japan Honshu Cropland"
        "Murray Darling Australia"
        "Western Australia Wheatbelt"
        "New Zealand Canterbury"
        "Central Asia Irrigated"
        "Spain Portugal Mediterranean"
        "Greece Balkans Agriculture"
        "Peru Coastal Irrigation"
    ];

    LatMin = [
        34; 37; 32; 30; 19; -2; -20; -30; -38; -38;
        38; 44; 45; 37; 22; 30; 5; 6; -5; -31;
        24; 22; 24; 12; 21; 8; 12; -8; 32; 28;
        40; 35; 34; -36; -34; -45; 39; 37; 39; -15
    ];

    LatMax = [
        40; 45; 41; 36; 23; 7; -5; -22; -30; -30;
        48; 46; 50; 42; 31; 36; 14; 14; 5; -24;
        31; 28; 32; 20; 27; 14; 18; -5; 40; 32;
        48; 40; 38; -29; -29; -41; 45; 43; 44; -5
    ];

    LonMin = [
        -122; -100; -103; -92; -103; -78; -60; -55; -65; -73;
        -5; 8; 28; 26; 25; -9; -5; 35; 30; 24;
        73; 80; 67; 74; 88; 104; 99; 106; 112; 102;
        120; 126; 136; 142; 115; 170; 60; -9; 20; -78
    ];

    LonMax = [
        -118; -84; -96; -88; -99; -72; -45; -48; -55; -70;
        6; 13; 38; 45; 33; 4; 10; 40; 40; 31;
        80; 90; 75; 80; 93; 108; 106; 112; 120; 108;
        128; 129; 141; 149; 120; 174; 72; 3; 27; -70
    ];

    BaseCloudProbability = [
        0.18; 0.45; 0.35; 0.45; 0.35; 0.60; 0.50; 0.45; 0.38; 0.35;
        0.45; 0.40; 0.42; 0.35; 0.15; 0.25; 0.55; 0.50; 0.55; 0.35;
        0.35; 0.45; 0.28; 0.38; 0.60; 0.65; 0.58; 0.70; 0.40; 0.55;
        0.42; 0.45; 0.50; 0.38; 0.30; 0.50; 0.30; 0.28; 0.36; 0.20
    ];

    % Sampling weights can be tuned later when real OPERA alert density is known.
    SampleWeight = ones(numel(Name),1);

    regions = table(Name, LatMin, LatMax, LonMin, LonMax, ...
        BaseCloudProbability, SampleWeight);

    regions.SampleWeight = regions.SampleWeight ./ sum(regions.SampleWeight);
end

function regionID = sampleRegionID()
    regions = agriculturalRegions();
    c = cumsum(regions.SampleWeight);
    regionID = find(rand() <= c, 1, 'first');
end

function [lat, lon, regionID, regionName, baseCloudProbability] = sampleAgriculturalLocation(regionID)
    regions = agriculturalRegions();

    if nargin < 1
        regionID = sampleRegionID();
    end

    lat = regions.LatMin(regionID) + ...
        (regions.LatMax(regionID) - regions.LatMin(regionID))*rand();

    lon = regions.LonMin(regionID) + ...
        (regions.LonMax(regionID) - regions.LonMin(regionID))*rand();

    regionName = regions.Name(regionID);
    baseCloudProbability = regions.BaseCloudProbability(regionID);
end

function [lat, lon, regionID, regionName, baseCloudProbability] = ...
    sampleAgriculturalLocationNear(regionID, centerLat, centerLon, spread_deg)

    regions = agriculturalRegions();

    lat = centerLat + spread_deg*randn();
    lon = centerLon + spread_deg*randn();

    lat = min(max(lat, regions.LatMin(regionID)), regions.LatMax(regionID));
    lon = min(max(lon, regions.LonMin(regionID)), regions.LonMax(regionID));

    regionName = regions.Name(regionID);
    baseCloudProbability = regions.BaseCloudProbability(regionID);
end

function poiCatalog = assignNDVIValues(poiCatalog, ndviLowThreshold, hlsEpsilon)
    N = height(poiCatalog);

    % Classes:
    %   1 = normal healthy vegetation
    %   2 = low-NDVI issue
    %   3 = HLS mismatch issue
    %   4 = both low-NDVI and HLS mismatch
    u = rand(N,1);
    class = ones(N,1);
    class(u >= 0.65 & u < 0.83) = 2;
    class(u >= 0.83 & u < 0.95) = 3;
    class(u >= 0.95) = 4;

    trueNDVI = zeros(N,1);
    hlsNDVI = zeros(N,1);

    for k = 1:N
        switch class(k)
            case 1
                trueNDVI(k) = clippedNormal(0.65, 0.08, 0.40, 0.90);
                hlsNDVI(k) = clip01(trueNDVI(k) + 0.04*randn());
            case 2
                trueNDVI(k) = clippedNormal(0.22, 0.07, 0.05, ndviLowThreshold - 0.02);
                hlsNDVI(k) = clip01(trueNDVI(k) + 0.04*randn());
            case 3
                trueNDVI(k) = clippedNormal(0.62, 0.08, 0.40, 0.90);
                mismatch = sign(randn()) * (hlsEpsilon + 0.08 + 0.05*rand());
                hlsNDVI(k) = clip01(trueNDVI(k) + mismatch);
            case 4
                trueNDVI(k) = clippedNormal(0.20, 0.06, 0.05, ndviLowThreshold - 0.02);
                mismatch = sign(randn()) * (hlsEpsilon + 0.08 + 0.05*rand());
                hlsNDVI(k) = clip01(trueNDVI(k) + mismatch);
        end
    end

    measuredNDVI = clip01(trueNDVI + 0.03*randn(N,1));

    poiCatalog.TrueNDVI = trueNDVI;
    poiCatalog.HLS_NDVI = hlsNDVI;
    poiCatalog.MeasuredNDVI_FirstObs = measuredNDVI;
    poiCatalog.SimulatedClass = class;
end

function lightingOK = generateLightingOK(poiCatalog, t_s, startDatetimeUTC, maxSolarZenith_deg)
    Npoi = height(poiCatalog);
    Nt = numel(t_s);

    lightingOK = false(Npoi, Nt);

    for p = 1:Npoi
        sza_deg = solarZenithAngle_deg( ...
            startDatetimeUTC, t_s, ...
            poiCatalog.Latitude_deg(p), poiCatalog.Longitude_deg(p));

        lightingOK(p,:) = (sza_deg <= maxSolarZenith_deg).';
    end
end

function sza_deg = solarZenithAngle_deg(startDatetimeUTC, t_s, lat_deg, lon_deg)
    % Approximate solar-zenith-angle model.
    % Good enough for trajectory/tasking simulation, not precision astronomy.

    timeVec = startDatetimeUTC + seconds(t_s);

    doy = day(timeVec, 'dayofyear');
    utcHour = hour(timeVec) + minute(timeVec)/60 + second(timeVec)/3600;

    B_deg = (360/365) .* (doy - 81);
    eqTime_min = 9.87*sind(2*B_deg) - 7.53*cosd(B_deg) - 1.5*sind(B_deg);

    decl_deg = 23.45 .* sind((360/365) .* (284 + doy));

    localSolarTime_hr = utcHour + lon_deg/15 + eqTime_min/60;
    hourAngle_deg = 15 .* (localSolarTime_hr - 12);

    cosZen = sind(lat_deg).*sind(decl_deg) + ...
             cosd(lat_deg).*cosd(decl_deg).*cosd(hourAngle_deg);

    cosZen = min(max(cosZen, -1), 1);
    sza_deg = acosd(cosZen);
end

function cloudBlocked = generateCloudBlockage( ...
    poiCatalog, t_s, totalSimDays, cloudinessScale, cloudDailySigma, ...
    cloudPersistence_hr, cloudProbabilityFloor, cloudProbabilityCeil)

    Npoi = height(poiCatalog);
    Nt = numel(t_s);

    cloudBlocked = false(Npoi, Nt);

    dt_hr = median(diff(t_s)) / 3600;
    persistence = exp(-dt_hr / cloudPersistence_hr);

    timeDay = floor(t_s/86400) + 1;
    timeDay(timeDay > totalSimDays) = totalSimDays;

    for p = 1:Npoi
        baseP = poiCatalog.BaseCloudProbability(p) * cloudinessScale;

        pCloudByDay = baseP + cloudDailySigma*randn(totalSimDays,1);
        pCloudByDay = min(max(pCloudByDay, cloudProbabilityFloor), cloudProbabilityCeil);

        stateCloudy = rand() < pCloudByDay(1);

        for k = 1:Nt
            d = timeDay(k);
            pCloud = pCloudByDay(d);

            if k > 1
                % Markov cloud model with approximate stationary cloud
                % probability pCloud and persistence set by cloudPersistence_hr.
                pClearToCloud = pCloud * (1 - persistence);
                pCloudToClear = (1 - pCloud) * (1 - persistence);

                if stateCloudy
                    stateCloudy = rand() >= pCloudToClear;
                else
                    stateCloudy = rand() < pClearToCloud;
                end
            end

            cloudBlocked(p,k) = stateCloudy;
        end
    end
end

function x = clippedNormal(mu, sigma, lo, hi)
    x = mu + sigma*randn();
    x = max(lo, min(hi, x));
end

function y = clip01(x)
    y = max(0, min(1, x));
end

function satDef = makeConstellation(Nsat, altitude_km, inclination_deg, architectureMode)
    satDef.Nsat = Nsat;
    satDef.altitude_km = altitude_km;
    satDef.inclination_deg = inclination_deg;
    satDef.architectureMode = architectureMode;

    raan_deg = zeros(Nsat,1);
    u0_deg = zeros(Nsat,1);
    planeID = zeros(Nsat,1);

    switch architectureMode
        case 'singlePlanePhased'
            % All satellites in one orbital plane, spaced along-track.
            satDef.numPlanes = 1;
            for s = 1:Nsat
                raan_deg(s) = 0;
                u0_deg(s) = 360*(s-1)/Nsat;
                planeID(s) = 1;
            end

        case 'balancedPlanes'
            % Simple Walker-like layout:
            % - Choose roughly sqrt(N) planes
            % - Space planes evenly in RAAN
            % - Space satellites evenly within each plane
            P = ceil(sqrt(Nsat));
            P = min(P, Nsat);
            satDef.numPlanes = P;

            basePerPlane = floor(Nsat/P);
            extra = mod(Nsat, P);

            s = 0;
            for p = 1:P
                satsThisPlane = basePerPlane + (p <= extra);
                thisRAAN = 360*(p-1)/P;

                for q = 1:satsThisPlane
                    s = s + 1;
                    raan_deg(s) = thisRAAN;
                    u0_deg(s) = 360*(q-1)/satsThisPlane;
                    planeID(s) = p;
                end
            end

        otherwise
            error('Unknown architectureMode: %s', architectureMode);
    end

    satDef.raan_deg = raan_deg;
    satDef.u0_deg = u0_deg;
    satDef.planeID = planeID;
end

function [latSS_deg, lonSS_deg] = propagateConstellationECEF( ...
    satDef, t_s, Re_km, mu_km3_s2, we_rad_s, J2, useJ2Precession)

    Nt = numel(t_s);
    Nsat = satDef.Nsat;

    latSS_deg = zeros(Nt, Nsat);
    lonSS_deg = zeros(Nt, Nsat);

    for s = 1:Nsat
        [latSS_deg(:,s), lonSS_deg(:,s)] = propagateCircularOrbitECEF( ...
            satDef.altitude_km, satDef.inclination_deg, satDef.raan_deg(s), ...
            satDef.u0_deg(s), t_s, Re_km, mu_km3_s2, we_rad_s, J2, useJ2Precession);
    end
end

function [latSS_deg, lonSS_deg] = propagateCircularOrbitECEF( ...
    altitude_km, inclination_deg, raan0_deg, u0_deg, t_s, Re_km, mu_km3_s2, ...
    we_rad_s, J2, useJ2Precession)

    a_km = Re_km + altitude_km;
    n_rad_s = sqrt(mu_km3_s2 / a_km^3);

    inc = deg2rad(inclination_deg);
    raan0 = deg2rad(raan0_deg);
    u0 = deg2rad(u0_deg);

    if useJ2Precession
        % Circular-orbit secular RAAN precession due to Earth's oblateness.
        raanDot_rad_s = -1.5 * J2 * (Re_km/a_km)^2 * n_rad_s * cos(inc);
    else
        raanDot_rad_s = 0;
    end

    u = u0 + n_rad_s * t_s;
    raan = raan0 + raanDot_rad_s * t_s;

    cosO = cos(raan);
    sinO = sin(raan);
    cosi = cos(inc);
    sini = sin(inc);

    xECI = a_km * (cosO.*cos(u) - sinO.*sin(u)*cosi);
    yECI = a_km * (sinO.*cos(u) + cosO.*sin(u)*cosi);
    zECI = a_km * (sin(u)*sini);

    theta = we_rad_s * t_s;
    xECEF =  cos(theta).*xECI + sin(theta).*yECI;
    yECEF = -sin(theta).*xECI + cos(theta).*yECI;
    zECEF =  zECI;

    rmag = sqrt(xECEF.^2 + yECEF.^2 + zECEF.^2);

    latSS_deg = asind(zECEF ./ rmag);
    lonSS_deg = atan2d(yECEF, xECEF);
end

function simOut = runPOISimulation( ...
    poiCatalog, latSS_deg, lonSS_deg, t_s, Re_km, poiAccessRadius_km, ...
    lightingOK, cloudBlocked, ndviLowThreshold, hlsEpsilon, ...
    fastResolveLimit_days, minSixBandRevisitDelay_hr)

    Npoi = height(poiCatalog);
    Nt = numel(t_s);
    Nsat = size(latSS_deg,2);

    fastResolveLimit_s = fastResolveLimit_days * 86400;
    minSixBandRevisitDelay_s = minSixBandRevisitDelay_hr * 3600;

    % Status codes:
    %   0 = not arrived yet
    %   1 = active, needs initial NDVI observation
    %   2 = suspicious, needs 6-band revisit
    %   3 = resolved/deleted
    status = zeros(Npoi,1);

    firstObsTime_s = NaN(Npoi,1);
    sixBandObsTime_s = NaN(Npoi,1);
    resolvedTime_s = NaN(Npoi,1);
    numNDVIObs = zeros(Npoi,1);
    numSixBandObs = zeros(Npoi,1);

    backlogCount = zeros(Nt,1);
    numMultiPOIImagingEvents = 0;

    for k = 1:Nt
        tNow = t_s(k);

        % Activate newly arrived POIs.
        newlyArrived = status == 0 & poiCatalog.ArrivalTime_s <= tNow;
        status(newlyArrived) = 1;

        activeIdx = find(status == 1 | status == 2);
        observedNow = false(Npoi,1);

        if ~isempty(activeIdx)
            for s = 1:Nsat
                d_km = greatCircleDistance_km( ...
                    latSS_deg(k,s), lonSS_deg(k,s), ...
                    poiCatalog.Latitude_deg(activeIdx), ...
                    poiCatalog.Longitude_deg(activeIdx), Re_km);

                geoAccessible = activeIdx(d_km <= poiAccessRadius_km);

                if ~isempty(geoAccessible)
                    valid = lightingOK(geoAccessible,k) & ~cloudBlocked(geoAccessible,k);
                    observedNow(geoAccessible(valid)) = true;
                end
            end
        end

        observedList = find(observedNow);

        if numel(observedList) > 1
            numMultiPOIImagingEvents = numMultiPOIImagingEvents + 1;
        end

        % Use status at the start of the timestep so one timestep cannot do
        % both first NDVI and six-band revisit.
        statusBefore = status;

        for ii = 1:numel(observedList)
            p = observedList(ii);

            if statusBefore(p) == 1
                % First observation: compute NDVI and compare against low
                % threshold and HLS-reported NDVI.
                firstObsTime_s(p) = tNow;
                numNDVIObs(p) = numNDVIObs(p) + 1;

                measuredNDVI = poiCatalog.MeasuredNDVI_FirstObs(p);
                hlsNDVI = poiCatalog.HLS_NDVI(p);

                suspicious = measuredNDVI < ndviLowThreshold || ...
                             abs(measuredNDVI - hlsNDVI) > hlsEpsilon;

                if suspicious
                    status(p) = 2;
                else
                    status(p) = 3;
                    resolvedTime_s(p) = tNow;
                end

            elseif statusBefore(p) == 2
                % Revisit: collect full 6-band image. After that, the POI
                % is resolved/deleted.
                if tNow >= firstObsTime_s(p) + minSixBandRevisitDelay_s
                    sixBandObsTime_s(p) = tNow;
                    numSixBandObs(p) = numSixBandObs(p) + 1;
                    status(p) = 3;
                    resolvedTime_s(p) = tNow;
                end
            end
        end

        backlogCount(k) = nnz(status == 1 | status == 2);
    end

    arrivedByEnd = poiCatalog.ArrivalTime_s <= t_s(end);
    totalArrived = nnz(arrivedByEnd);
    resolved = status == 3;

    resolutionTime_s = resolvedTime_s - poiCatalog.ArrivalTime_s;
    resolvedFast = resolved & resolutionTime_s < fastResolveLimit_s;

    resolutionTime_days = resolutionTime_s(resolved) / 86400;

    simOut.totalArrived = totalArrived;
    simOut.numResolved = nnz(resolved);
    simOut.resolutionFraction = simOut.numResolved / totalArrived;
    simOut.numResolvedFast = nnz(resolvedFast);
    simOut.fastResolutionFraction = simOut.numResolvedFast / totalArrived;
    simOut.numResolvedImmediate = nnz(resolved & numSixBandObs == 0);
    simOut.numResolvedAfterSixBand = nnz(resolved & numSixBandObs > 0);
    simOut.numUnresolvedEnd = nnz(status == 1 | status == 2);

    if isempty(resolutionTime_days)
        simOut.meanResolutionTime_days = NaN;
        simOut.medianResolutionTime_days = NaN;
    else
        simOut.meanResolutionTime_days = mean(resolutionTime_days);
        simOut.medianResolutionTime_days = median(resolutionTime_days);
    end

    simOut.meanBacklog = mean(backlogCount);
    simOut.backlogCount = backlogCount;
    simOut.backlogTime_days = t_s / 86400;
    simOut.finalStatus = status;
    simOut.firstObsTime_s = firstObsTime_s;
    simOut.sixBandObsTime_s = sixBandObsTime_s;
    simOut.resolvedTime_s = resolvedTime_s;
    simOut.numNDVIObs = numNDVIObs;
    simOut.numSixBandObs = numSixBandObs;
    simOut.numMultiPOIImagingEvents = numMultiPOIImagingEvents;
end

function d_km = greatCircleDistance_km(lat1_deg, lon1_deg, lat2_deg, lon2_deg, Re_km)
    % Vectorized great-circle distance from one point to many points.
    lat1 = deg2rad(lat1_deg);
    lon1 = deg2rad(lon1_deg);
    lat2 = deg2rad(lat2_deg);
    lon2 = deg2rad(lon2_deg);

    dlat = lat2 - lat1;
    dlon = lon2 - lon1;

    a = sin(dlat/2).^2 + cos(lat1).*cos(lat2).*sin(dlon/2).^2;
    c = 2*atan2(sqrt(a), sqrt(1-a));

    d_km = Re_km * c;
end

function idx = findScenario(allScenarioOutputs, architectureMode, Nsat)
    idx = [];
    for k = 1:numel(allScenarioOutputs)
        if strcmp(allScenarioOutputs(k).architectureMode, architectureMode) && ...
           allScenarioOutputs(k).Nsat == Nsat
            idx = k;
            return;
        end
    end
    error('Scenario not found.');
end

function [lonPlot_deg, latPlot_deg] = breakTrackAtDateLine(lon_deg, lat_deg)
    lonPlot_deg = lon_deg;
    latPlot_deg = lat_deg;
    jumpIdx = find(abs(diff(lon_deg)) > 180);
    lonPlot_deg(jumpIdx + 1) = NaN;
    latPlot_deg(jumpIdx + 1) = NaN;
end

function [circleLat_deg, circleLon_deg] = smallCircleLatLon(centerLat_deg, centerLon_deg, radius_km, Re_km)
    % Small circle around a lat/lon point.
    bearings = linspace(0, 2*pi, 181).';
    delta = radius_km / Re_km;

    lat1 = deg2rad(centerLat_deg);
    lon1 = deg2rad(centerLon_deg);

    lat2 = asin(sin(lat1)*cos(delta) + cos(lat1)*sin(delta).*cos(bearings));
    lon2 = lon1 + atan2(sin(bearings).*sin(delta).*cos(lat1), ...
                        cos(delta) - sin(lat1).*sin(lat2));

    circleLat_deg = rad2deg(lat2);
    circleLon_deg = wrapTo180Local(rad2deg(lon2));
end

function lonWrapped = wrapTo180Local(lon)
    lonWrapped = mod(lon + 180, 360) - 180;
end
