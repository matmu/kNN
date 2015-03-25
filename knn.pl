#!/usr/bin/perl

use strict;
use warnings;
use List::Util qw(sum);
use Getopt::Long;

my $input_file;
my $output_file;
my $training_file;
my $split_ratio = 0.66;
my $normalize = 0;
my $k;

GetOptions(
	'i|input_file=s' => \$input_file, 
	'o|output_file=s' => \$output_file,
	'n|normalize' => \$normalize,
	't|training_file=s' => \$training_file,
	's|split_ratio=f' => \$split_ratio,
	'k=i'=> \$k
) or die "Invalid parameters!";


print "Input file: ".$input_file."\n";
print "Output_file: ".$output_file."\n" if (defined $output_file);
print "Training file: ".$training_file."\n";
print "Split ratio: ".$split_ratio."\n";
print "Normalize: ".$normalize."\n";
print "k: ".$k."\n" if (defined $k);

my @input = _load_data($input_file);
my @training = _load_data($training_file);
print "Size of training and prediction set: ".@training."/".@input."\n";

my $l = @{$training[0]}-1;
print "Number of features in training set: $l\n";

if ($normalize){
	_normalize([@training, @input], $l);
}

#foreach (@training){
#	foreach my $a (@{$_}){
#		print $a."\t";
#	}
#	print "\n";
#}
#die;

if (!defined $k){
	print "Find best k...\n";
	$k = _best_k(\@training, $split_ratio, $l, 10, 5);
}


foreach my $t (@input){
	my @neighbors = _get_neighbours(\@training, $t, 4, $k);
	my $response = _get_response(\@neighbors);
	
	print join("\t", (@{$t}[0..$l-1], $response))."\n";
	
	# print "Predicted: $response, Actual: ".${$t}[@{$t}-1]."\n";
}
# $accuracy = sprintf("%.2f", $accuracy);

sub _normalize {
	my ($records) = @_;
	
	my @max;
	my @min;
	
	for (my $i=0; $i<@{$records}; $i++){
		my $rec = ${$records}[$i];
		
		for (my $j=0; $j<@{$rec}-1; $j++){
			my $val = ${$rec}[$j];
			if (!defined $max[$j] || $val > $max[$j]){
				$max[$j] = $val;
			}
			if (!defined $min[$j] || $val < $min[$j]){
				$min[$j] = $val;
			}
		}
	}
	
	
	for (my $i=0; $i<@{$records}; $i++){
		my $rec = ${$records}[$i];
		
		for (my $j=0; $j<@{$rec}-1; $j++){
			my $val = ${$rec}[$j];
			my $max = $max[$j];
			my $min = $min[$j];
			
			if ($min >= 0){
				if($max - $min==0){
					print $max, $min,"\n";
				}
				${$rec}[$j] = ($val - $min)/($max - $min);
			}
			else {
				${$rec}[$j] = ($val + $min)/($max + $min);
			}
		}
	}
	
	return 1;
}


sub _best_k {
	my ($records, $split_ratio, $l, $runs, $stop_after) = @_;
	
	my $best_k;
	my $best_accuracy = 0;
	
	my $not_better = 0;
	my $k = 0;
	
	while($not_better < $stop_after){
		$k++;
	
		my @accuracies;
		for (my $i=1; $i<$runs; $i++){
			
			my ($training_set, $test_set) = _split_records($records, $split_ratio);
			
			my @predictions;
			foreach my $t (@{$test_set}){
				my @neighbors = _get_neighbours($training_set, $t, $l, $k);
				my $response = _get_response(\@neighbors);
				push(@predictions, $response);
			}
			
			my $accuracy = _get_accuracy($test_set, \@predictions);
			push(@accuracies, $accuracy);
		}
		
		my $mean_accuracy = sum(@accuracies)/@accuracies;
		print "k: $k; Mean Accuracy: $mean_accuracy\n";
		@accuracies = ();
		
		if ($mean_accuracy > $best_accuracy){
			$best_k = $k;
			$best_accuracy = $mean_accuracy;
			$not_better = 0;
		}
		else {
			$not_better++;
		}
	}
	
	print "Choose k=$best_k (Mean Accuracy: $best_accuracy)\n";
	
	return $best_k;
}


sub _load_data {
	my ($input_file) = @_;
	
	my @records;
	open(IN, "<$input_file") or die "Cannot open file: $!\n";
	while(<IN>){
		chomp($_);
		my @record = split(/\t/, $_);
		#my $class = pop @record;
		push(@records, \@record);
	}
	close IN;
	
	return @records;
}


sub _split_records {
	my ($records, $split) = @_;
	
	my (@training_set, @test_set);
	foreach (@{$records}){
		if (rand() < $split){
			push(@training_set, $_);
		}
		else {
			push(@test_set, $_);
		}
	}
	
	return (\@training_set, \@test_set);
}


#my $test_data1 = [1,2,3,4];
#my $test_data2 = [1,2,3,32];
#print _euclidean_distance($test_data1, $test_data2);
sub _euclidean_distance {
	my ($features1, $features2, $l) = @_;
	
	my $distance = 0;
	for (my $i=0; $i<$l; $i++){
		$distance += (${$features1}[$i] - ${$features2}[$i])**2;
	}
	
	return $distance**(0.5);
}


sub _get_neighbours {
	my ($training_set, $test_instance, $l, $k) = @_;
	
	my %dist2neighbours;
	
	foreach my $t (@{$training_set}){
		my $dist = _euclidean_distance($t, $test_instance, $l);
		push(@{$dist2neighbours{$dist}}, $t);
	}
	
	 my @sorted_keys = sort {$a <=> $b} keys(%dist2neighbours);
	
	my @neighbours;
	while(@neighbours < $k){
		 push(@neighbours, @{$dist2neighbours{shift @sorted_keys}});
	}
	
	while (@neighbours > $k){
		pop @neighbours;
	}
	
	return @neighbours;
}


sub _get_response {
	my ($neighbors) = @_;
	
	my %class2votes;
	foreach my $n (@{$neighbors}){
		
		my $class = ${$n}[@{$n}-1];
		
		if (exists($class2votes{$class})){
			$class2votes{$class}++;
		}
		else {
			$class2votes{$class} = 1;
		}
	}
	
	my @sorted_keys = sort {$class2votes{$b} <=> $class2votes{$a} } keys(%class2votes);
	
	return $sorted_keys[0];
}


sub _get_accuracy {
	my ($test_set, $predictions) = @_;
	
	die if (@{$test_set} != @{$predictions});
	
	my $correct = 0;
	for(my $i=0; $i<@{$test_set}; $i++){
		if (${$test_set}[$i][@{${$test_set}[$i]}-1] eq ${$predictions}[$i]){
			$correct++;
		}
	}
	return $correct/@{$test_set};
}
